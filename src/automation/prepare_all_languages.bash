set -euo pipefail

BASE_DIR=${1-}
DEVICE=${2-}

# LANGUAGE DATASET PREPARATION SCRIPTS
prepare_regular_script=recognizers/string_sampling/prepare_automaton_dataset.bash

prepare_non_regular_script=recognizers/string_sampling/prepare_hand_coded_dataset.bash

prepare_test_edit_script=recognizers/string_sampling/prepare_test_edit_distance.bash

# LANGUAGES DATASETS
regular_languages=(even-pairs repeat-01 parity cycle-navigation modular-arithmetic-simple dyck-2-3 first)

non_regular_languages=(majority stack-manipulation marked-reversal unmarked-reversal marked-copy missing-duplicate-string odds-first binary-addition binary-multiplication compute-sqrt bucket-sort)

test_edit_languages=(repeat-01 dyck-2-3)

all_languages=("${regular_languages[@]}" "${non_regular_languages[@]}")

# LANGUAGE DATASETS PREPARATION

for lang in ${regular_languages[@]}; do
    echo ""
    echo "PREP RG: $lang"
    echo "COMMAND: bash $prepare_regular_script $BASE_DIR $lang $DEVICE"
    bash $prepare_regular_script $BASE_DIR $lang $DEVICE
done

for lang in ${non_regular_languages[@]}; do
    echo ""
    echo "PREP NR: $lang"
    bash $prepare_non_regular_script $BASE_DIR $lang $DEVICE
done

for lang in ${test_edit_languages[@]}; do
    echo ""
    echo "PREP TE: $lang"
    echo "COMMAND: bash $prepare_test_edit_script $BASE_DIR $lang $DEVICE"
    bash $prepare_test_edit_script $BASE_DIR $lang $DEVICE
done
