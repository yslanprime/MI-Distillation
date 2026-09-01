#!/usr/bin/env bash
# Stage 3 (optional): build one training set per fixed lambda, covering the
# Short CoT (lambda=1.0) and Long CoT (lambda=0.0) endpoints.
#
# Not required by MI-Distillation, which filters internally in stage 4.
#
# Usage:
#   TOKENIZER_MODEL=pretrained_model/Qwen2.5-3B-Instruct \
#     bash scripts/03_build_fixed_lambda_data.sh

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

POOL_DIR="${POOL_DIR:-${COT_DIR}/${DATASET_NAME}_${TEACHER_TAG}}"
FIXED_FILTER_DIR="${FIXED_FILTER_DIR:-${FILTER_DIR}/${DATASET_NAME}_${TEACHER_TAG}}"
FIXED_TRAIN_DIR="${FIXED_TRAIN_DIR:-${TRAIN_DATA_DIR}}"

require_path "${POOL_DIR}" "Run scripts/02_generate_cot.sh first."
require_path "${TOKENIZER_MODEL}" "Set TOKENIZER_MODEL to the student tokenizer used for length filtering."

banner "Stage 3: fixed-lambda training sets"
echo "  Candidate pool : ${POOL_DIR}"
echo "  Tokenizer      : ${TOKENIZER_MODEL}"
echo "  Filtered dir   : ${FIXED_FILTER_DIR}"
echo "  Train data dir : ${FIXED_TRAIN_DIR}"
echo "  Train / val    : ${TRAIN_SIZE} / ${VAL_SIZE}"
echo "  Max length     : < ${MAX_LENGTH}"

"${PYTHON_BIN}" src/filter_dataset.py \
  --input_dir   "${POOL_DIR}" \
  --input_glob  "*.json" \
  --output_dir  "${FIXED_FILTER_DIR}" \
  --model_path  "${TOKENIZER_MODEL}" \
  --instruction "${EVAL_INSTRUCTION}" \
  --max_length  "${MAX_LENGTH}" \
  --train_size  "${TRAIN_SIZE}" \
  --val_size    "${VAL_SIZE}" \
  --seed        "${SEED}"

echo ""
echo "Exporting to LLaMA-Factory format ..."
mkdir -p "${FIXED_TRAIN_DIR}"

shopt -s nullglob
filtered_files=("${FIXED_FILTER_DIR}"/*_train"${TRAIN_SIZE}".json)
shopt -u nullglob

if ((${#filtered_files[@]} == 0)); then
  echo "[ERROR] No *_train${TRAIN_SIZE}.json files found under ${FIXED_FILTER_DIR}." >&2
  exit 1
fi

for input_file in "${filtered_files[@]}"; do
  dataset_name="$(basename "${input_file}" .json)"
  output_file="${FIXED_TRAIN_DIR}/${dataset_name}_llamafactory.json"

  echo "  ${input_file} -> ${output_file}"
  "${PYTHON_BIN}" src/export_llamafactory.py \
    --input_file   "${input_file}" \
    --output_file  "${output_file}" \
    --instruction  "${EVAL_INSTRUCTION}" \
    --dataset_info "${DATASET_INFO}" \
    --dataset_name "${dataset_name}"
done

banner "Stage 3 complete. Datasets registered in ${DATASET_INFO}"
