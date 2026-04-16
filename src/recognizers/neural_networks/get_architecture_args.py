import argparse
import math
import pathlib

from recognizers.neural_networks.data import load_vocabulary_data

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', choices=['transformer', 'rnn', 'lstm', 'gru', 'synced_difflogic'], required=True)
    parser.add_argument('--parameter-budget', type=int, required=True)
    parser.add_argument('--training-data', type=pathlib.Path, required=True)
    args = parser.parse_args()

    vocab = load_vocabulary_data(args, parser)
    vocab_size = len(vocab.tokens) + int(vocab.allow_unk)
    num_layers = 5

    outputs = [
        '--architecture', args.architecture,
        '--num-layers', str(num_layers)
    ]

    if args.architecture == 'transformer':
        # num_params =
        #   vocab_size * d_model +    # embeddings
        #   num_layers * (
        #       d_model * 4 * (d_model + 1) +    # projection layers
        #       2 * feedforward_size * (d_model + 1) +    # feedforward layer
        #       4 * d_model    # layer norm
        #   ) +
        #   2 * d_model +    # final layer norm
        #   d_model + 1    # recognition head
        # d_model = size * num_heads
        # feedforward_size = feedforward_size_factor * d_model
        num_heads = 8
        feedforward_size_factor = 4
        a = num_layers * (4 + 2 * feedforward_size_factor)
        b = vocab_size + num_layers * (8 + 2 * feedforward_size_factor) + 3
        c = 1 - args.parameter_budget
        d_model_float = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
        size_float = d_model_float / num_heads
        size = round(size_float)
        d_model = size * num_heads
        feedforward_size = d_model * feedforward_size_factor
        outputs.extend([
            '--d-model', str(d_model),
            '--num-heads', str(num_heads),
            '--feedforward-size', str(feedforward_size)
        ])
    elif args.architecture in ('rnn', 'lstm', 'gru'):
        # RNN:
        # num_params =
        #   vocab_size * hidden_units +    # embeddings
        #   num_layers * hidden_units +    # initial hidden state
        #   num_layers * (
        #       hidden_units * (2 * hidden_units + 1)    # input/recurrent layers
        #   ) +
        #   hidden_units + 1    # recognition head
        # LSTM:
        # num_params =
        #   vocab_size * hidden_units +    # embeddings
        #   num_layers * hidden_units +    # initial hidden state
        #   num_layers * (
        #       4 * hidden_units * (2 * hidden_units + 1)    # input/recurrent layers
        #   ) +
        #   hidden_units + 1    # recognition head
        if args.architecture == 'rnn':
            a = 2 * num_layers
            b = vocab_size + 2 * num_layers + 1
        elif args.architecture == 'gru':
            a = 6 * num_layers
            b = vocab_size + 3 * num_layers + 1
        else:
            a = 8 * num_layers
            b = vocab_size + 5 * num_layers + 1
        c = 1 - args.parameter_budget
        hidden_units_float = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
        hidden_units = round(hidden_units_float)
        outputs.extend([
            '--hidden-units', str(hidden_units)
        ])
    elif args.architecture == 'synced_difflogic':
        # num_params =
        #   vocab_size * embedding_dim +              # embedding
        #   num_layers * hidden_units * 16 +          # N-layers (each LogicLayer has out_dim*16 weights)
        #   num_layers * hidden_units * 16 +          # K-layers
        #   (num_layers + group_factor) * hidden_units * 16 +  # M-layers (num_layers uniform + 1 final of size hidden_units*group_factor)
        #   hidden_units + 1                          # recognition head Linear(hidden_units, 1)
        # = vocab_size * embedding_dim + 16 * hidden_units * (3 * num_layers + group_factor) + hidden_units + 1
        #
        # Strategy: set embedding_dim = hidden_units (satisfies constraint embedding_dim <= 2*hidden_units).
        # Substituting gives a linear equation in hidden_units:
        #   num_params = hidden_units * (vocab_size + 16 * (3 * num_layers + group_factor) + 1) + 1
        #   hidden_units = round((budget - 1) / (vocab_size + 16 * (3 * num_layers + group_factor) + 1))
        group_factor = 2  # matches model_interface.py default

        coeff = vocab_size + 16 * (3 * num_layers + group_factor) + 1
        hidden_units = round((args.parameter_budget - 1) / coeff)
        embedding_dim = hidden_units  # E = H satisfies the E <= 2*H constraint

        # coeff = 16 * (3 * num_layers + group_factor) + 1
        # hidden_units = round((args.parameter_budget - vocab_size * vocab_size - 1) / coeff)
        # embedding_dim = vocab_size

        outputs.extend([
            '--hidden-units', str(hidden_units),
            '--embedding-dim', str(embedding_dim)
        ])
    else:
        raise NotImplementedError
    outputs.extend(['--dropout', '0.1'])
    print(' '.join(outputs))

if __name__ == '__main__':
    main()
