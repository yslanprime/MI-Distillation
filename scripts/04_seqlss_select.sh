#!/usr/bin/env bash
# Stage 4: MI-Distillation data construction via SeqLSS selection.
#
# SeqLSS is student-conditioned, so rerun this for every student you distill
# into. Sweep ALPHAS to vary the learnability penalty.
#
# Usage:
#   STUDENT_MODEL=pretrained_model/Qwen2.5-3B-Instruct bash scripts/04_seqlss_select.sh
#   STUDENT_MODEL=pretrained_model/Llama-3.2-3B-Instruct ALPHAS="1 2 3 4" \
#     bash scripts/04_seqlss_select.sh

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

STUDENT_MODEL="${STUDENT_MODEL:-${PRETRAINED_DIR}/Qwen2.5-3B-Instruct}"
POOL_DIR="${POOL_DIR:-${COT_DIR}/${DATASET_NAME}_${TEACHER_TAG}}"
ALPHAS="${ALPHAS:-${ALPHA}}"
GROUP_KEY="${GROUP_KEY:-unique_id}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
SAVE_ALL_VALID="${SAVE_ALL_VALID:-0}"

require_path "${POOL_DIR}" "Run scripts/02_generate_cot.sh first."
require_path "${STUDENT_MODEL}" "Set STUDENT_MODEL to the student you will distill into."

banner "Stage 4: SeqLSS selection"
echo "  Candidate pool : ${POOL_DIR}"
echo "  Student model  : ${STUDENT_MODEL}"
echo "  Alphas         : ${ALPHAS}"
echo "  Train size     : ${TRAIN_SIZE}"
echo "  Max length     : < ${MAX_LENGTH}"
echo "  Group key      : ${GROUP_KEY}"
echo "  Dataset info   : ${DATASET_INFO}"

for alpha in ${ALPHAS}; do
  echo ""
  echo "------------------------------------------------------------"
  echo " alpha = ${alpha}"
  echo "------------------------------------------------------------"

  args=(
    --input_dir       "${POOL_DIR}"
    --input_glob      "*.json"
    --model_path      "${STUDENT_MODEL}"
    --output_dir_root "${DATA_DIR}"
    --alpha           "${alpha}"
    --instruction     "${EVAL_INSTRUCTION}"
    --train_size      "${TRAIN_SIZE}"
    --max_length      "${MAX_LENGTH}"
    --seed            "${SEED}"
    --group_key       "${GROUP_KEY}"
    --dtype           "${DTYPE}"
    --device_map      "${DEVICE_MAP}"
    --register_llamafactory
    --dataset_info_path "${DATASET_INFO}"
  )
  if [[ "${SAVE_ALL_VALID}" == "1" ]]; then
    args+=(--save_all_valid)
  fi

  "${PYTHON_BIN}" src/seqlss_selection.py "${args[@]}"
done

banner "Stage 4 complete. Datasets registered in ${DATASET_INFO}"
