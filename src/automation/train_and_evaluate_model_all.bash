set -euo pipefail

BASE_DIR=${1-}
ARCHITECTURE=${2-}
LOSS_TERM=${3-}

train_and_eval_script=recognizers/neural_networks/train_and_evaluate.bash

validation_sets=(validation-short validation-long)
trials=({1..10})

regular_languages=(even-pairs repeat-01 parity cycle-navigation modular-arithmetic-simple dyck-2-3 first)

non_regular_languages=(majority stack-manipulation marked-reversal unmarked-reversal marked-copy missing-duplicate-string odds-first binary-addition binary-multiplication compute-sqrt bucket-sort)

languages=("${regular_languages[@]}" "${non_regular_languages[@]}")

for language in "${languages[@]}"; do
    for validation_data in "${validation_sets[@]}"; do
        for trial in "${trials[@]}"; do
            echo ""
            echo "TRAINING AND EVALUATING: $validation_data $trial"
            echo "COMMAND: bash $train_and_eval_script $BASE_DIR $language $ARCHITECTURE $LOSS_TERM $validation_data $trial"
            bash $train_and_eval_script $BASE_DIR $language $ARCHITECTURE $LOSS_TERM $validation_data $trial
        done
    done
done
