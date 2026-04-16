set -euo pipefail

BASE_DIR=${1-}
LANGUAGE=${2-}
ARCHITECTURE=${3-}
LOSS_TERM=${4-}

train_and_eval_script=recognizers/neural_networks/train_and_evaluate.bash

validation_sets=(validation-short validation-long)
trials=({1..10})

for validation_data in "${validation_sets[@]}"; do
    for trial in "${trials[@]}"; do
        echo ""
        echo "TRAINING AND EVALUATING: $validation_data $trial"
        echo "COMMAND: bash $train_and_eval_script $BASE_DIR $LANGUAGE $ARCHITECTURE $LOSS_TERM $validation_data $trial"
        bash $train_and_eval_script $BASE_DIR $LANGUAGE $ARCHITECTURE $LOSS_TERM $validation_data $trial
    done
done
