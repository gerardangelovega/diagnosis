import argparse
import logging
import sys
import collections
import json
import math
import pathlib
import torch

import humanfriendly

from rau.tools.torch.profile import get_current_memory

from recognizers.neural_networks.data import (
    add_data_arguments,
    load_prepared_data,
    load_vocabulary_data
)
from recognizers.neural_networks.model_interface import RecognitionModelInterface
from recognizers.neural_networks.training_loop import (
    RecognitionTrainingLoop,
    add_training_loop_arguments,
    get_training_loop_kwargs,
)

from rau.tasks.common.training_loop import MicroAveragedScoreAccumulator

from recognizers.tools.jsonl import write_json_line
from recognizers.neural_networks.data import load_prepared_data_from_directory
from recognizers.neural_networks.training_loop import generate_batches, get_loss_terms

def evaluate(model, model_interface, batches, num_examples, eval_mode='soft'):
    device = model_interface.get_device(None)
    example_scores = [None] * num_examples
    example_predictions = [None] * num_examples
    model.eval()
    if eval_mode == 'discrete':
        model.inner.set_mode('eval_col_all')
    with torch.inference_mode():
        for indexed_batch in batches:
            batch = [(x, d) for x, (i, d) in indexed_batch]
            prepared_batch = model_interface.prepare_batch(batch, device)
            batch_score_dict = get_loss_terms(
                model,
                model_interface,
                prepared_batch,
                numerator_reduction='none',
                denominator_reduction='none',
                label_smoothing_factor=0.0,
                include_accuracy=True
            )
            # Extract per-example predictions before passing to split_score_dict.
            batch_predictions = batch_score_dict.pop('recognition_prediction')[0].tolist()
            example_score_dicts = split_score_dict(batch, batch_score_dict)
            for (x, (i, d)), example_score_dict, prediction in zip(indexed_batch, example_score_dicts, batch_predictions):
                example_scores[i] = example_score_dict
                example_predictions[i] = prediction
    return example_scores, example_predictions

class DictScoreAccumulator:

    def __init__(self):
        super().__init__()
        self.scores = collections.defaultdict(MicroAveragedScoreAccumulator)

    def update(self, scores: dict[str, tuple[float, float]]) -> None:
        for key, (numerator, denominator) in scores.items():
            self.scores[key].update(numerator, denominator)

    def get_value(self) -> dict[str, float]:
        return { k : v.get_value() for k, v in self.scores.items() }

def split_score_dict(batch, batch_score_dict):
    batch_score_dict = {
        k : (n.tolist(), d.tolist() if d is not None else d)
        for k, (n, d) in batch_score_dict.items()
    }
    positive_index = 0
    for index, example in enumerate(batch):
        label = example[1][0]
        example_score_dict = {}
        for key, (numerator, denominator) in batch_score_dict.items():
            if not isinstance(numerator, list):
                # batch-level scalar metric like binary_reg has no
                # per-example breakdown skip it in per-example output.
                continue
            if len(numerator) < len(batch):
                if label:
                    example_score_dict[key] = (
                        numerator[positive_index],
                        denominator[positive_index] if denominator is not None else 1
                    )
            else:
                example_score_dict[key] = (
                    numerator[index],
                    denominator[index] if denominator is not None else 1
                )
        yield example_score_dict
        positive_index += int(label)

def main():

    # Configure logging to stdout.
    console_logger = logging.getLogger('main')
    console_logger.addHandler(logging.StreamHandler(sys.stdout))
    console_logger.setLevel(logging.INFO)
    console_logger.info(f'arguments: {sys.argv}')

    model_interface = RecognitionModelInterface()

    # Parse command-line arguments.
    parser = argparse.ArgumentParser(
        description=
        'Train a recognizer.'
    )
    parser.add_argument('--training-data', type=pathlib.Path, required=True,
        help='A directory containing training data. The file '
             '<training-data>/datasets/<input>/main.prepared will be used as '
             'input, and the file '
             '<training-data>/main.vocab will be used as the vocabulary.')
    parser.add_argument('--datasets', nargs='+', required=True,
        help='Names of datasets in the training data directory that will be '
             'used as input. The file '
             '<training-data>/datasets/<dataset>/main.prepared will be used as '
             'input. Multiple datasets can be passed. The name "training" '
             'can be used to evaluate on the training data.')
    parser.add_argument('--output', type=pathlib.Path, required=True,
        help='A directory where output files will be written.')
    parser.add_argument('--batching-max-tokens', type=int, required=True,
        help='The maximum number of tokens allowed per batch.')
    parser.add_argument('--eval-mode', choices=['soft', 'discrete'], default='soft',
        help='Evaluation mode. "soft" (default) uses soft logic. "discrete" fully '
             'discretizes the model via eval_col_all (synced_difflogic only; '
             'silently skipped for other architectures).')
    add_data_arguments(parser)
    model_interface.add_arguments(parser)
    model_interface.add_forward_arguments(parser)
    add_training_loop_arguments(parser)
    args = parser.parse_args()
    console_logger.info(f'parsed arguments: {args}')

    # Are we training on CPU or GPU?
    device = model_interface.get_device(args)
    console_logger.info(f'device: {device}')
    do_profile_memory = device.type == 'cuda'

    # Configure the training loop.
    training_loop = RecognitionTrainingLoop(
        **get_training_loop_kwargs(parser, args)
    )

    # Load the tokens in the vocabulary. This determines the sizes of the
    # embedding and softmax layers in the model.
    vocabulary_data = load_vocabulary_data(args, parser)

    if do_profile_memory:
        memory_before = get_current_memory(device)
    # Construct the model.
    saver = model_interface.construct_saver(args, vocabulary_data)
    # Log some information about the model: parameter random seed, number of
    # parameters, GPU memory.
    if model_interface.parameter_seed is not None:
        console_logger.info(f'parameter random seed: {model_interface.parameter_seed}')
    num_parameters = sum(p.numel() for p in saver.model.parameters())
    console_logger.info(f'number of parameters: {num_parameters}')
    if do_profile_memory:
        model_size_in_bytes = get_current_memory(device) - memory_before
        console_logger.info(f'model size: {humanfriendly.format_size(model_size_in_bytes)}')
    else:
        model_size_in_bytes = None

    # Load the data.
    training_data, validation_data, vocabulary \
        = load_prepared_data(args, parser, vocabulary_data, model_interface)

    # Start logging events to disk.
    with saver.logger() as event_logger:
        event_logger.log('model_info', dict(
            parameter_seed=model_interface.parameter_seed,
            size_in_bytes=model_size_in_bytes,
            num_parameters=num_parameters
        ))
        event_logger.log('training_info', dict(
            max_tokens_per_batch=args.max_tokens_per_batch,
            language_modeling_loss_coefficient=args.language_modeling_loss_coefficient,
            next_symbols_loss_coefficient=args.next_symbols_loss_coefficient
        ))
        # Run the training loop.
        training_loop.run(
            saver,
            model_interface,
            training_data,
            validation_data,
            vocabulary,
            console_logger,
            event_logger
        )

    model_interface.add_arguments(parser)
    model_interface.add_forward_arguments(parser)
    args = parser.parse_args()

    saver = model_interface.construct_saver(args)
    if args.eval_mode == 'discrete':
        if model_interface.architecture != 'synced_difflogic':
            print(
                f'info: --eval-mode discrete is not supported for architecture '
                f'{model_interface.architecture!r}; skipping.',
                file=sys.stderr
            )
            sys.exit(0)
    for dataset in args.datasets:
        if dataset == 'training':
            input_directory = args.training_data
        else:
            input_directory = args.training_data / 'datasets' / dataset
        examples = load_prepared_data_from_directory(
            input_directory,
            model_interface
        )
        examples = [(x, (i, d)) for i, (x, d) in enumerate(examples)]
        batches = generate_batches(examples, args.batching_max_tokens)
        scores, predictions = evaluate(saver.model, model_interface, batches, len(examples), eval_mode=args.eval_mode)
        accumulator = DictScoreAccumulator()
        example_scores_path = args.output / f'{dataset}.jsonl'
        print(f'writing {example_scores_path}')
        with example_scores_path.open('w') as fout:
            for score_dict in scores:
                write_json_line(score_dict, fout)
                accumulator.update(score_dict)
        predictions_path = args.output / f'{dataset}_predictions.jsonl'
        print(f'writing {predictions_path}')
        with predictions_path.open('w') as fout:
            for prediction in predictions:
                write_json_line({'prediction': prediction}, fout)
        total_scores = accumulator.get_value()
        total_scores_path = args.output / f'{dataset}.json'
        print(f'writing {total_scores_path}')
        with total_scores_path.open('w') as fout:
            json.dump(dict(scores=total_scores), fout, indent=2)
        json.dump(total_scores, sys.stdout, indent=2)
        print()

if __name__ == '__main__':
    main()
