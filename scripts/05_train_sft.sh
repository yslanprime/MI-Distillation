#!/usr/bin/env bash
# Stage 5: distill a student with supervised fine-tuning.
#
# Wraps llamafactory-cli with the paper's hyperparameters. DATASET must already
# be registered in ${DATASET_INFO} by stage 3 or 4.
#
# Usage:
#   STUDENT=qwen3b DATASET=<registered-name> RUN_NAME=<run-dir> \
#     bash scripts/05_train_sft.sh

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

STUDENT="${STUDENT:-qwen3b}"
DATASET="${DATASET:-}"
RUN_NAME="${RUN_NAME:-}"
INIT_FROM="${INIT_FROM:-}"          # optional: continue from an existing checkpoint
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed/ds_z3_offload_config.json}"

if [[ -z "${DATASET}" ]]; then
  echo "[ERROR] DATASET is required (a name registered in ${DATASET_INFO})." >&2
  exit 1
fi

# Student presets: model path plus the matching LLaMA-Factory chat template.
case "${STUDENT}" in
  qwen3b)  MODEL_PATH="${PRETRAINED_DIR}/Qwen2.5-3B-Instruct";   TRAIN_YAML="configs/sft/sft_qwen3b.yaml"  ;;
  llama3b) MODEL_PATH="${PRETRAINED_DIR}/Llama-3.2-3B-Instruct"; TRAIN_YAML="configs/sft/sft_llama3b.yaml" ;;
  *)
    echo "[ERROR] Unknown STUDENT: ${STUDENT}" >&2
    echo "        Valid values: qwen3b, llama3b" >&2
    echo "        To add another student, create configs/sft/sft_<name>.yaml with the" >&2
    echo "        matching LLaMA-Factory chat template and extend this case block." >&2
    exit 1
    ;;
esac

# Training resumes from INIT_FROM when set, otherwise from the base student.
if [[ -n "${INIT_FROM}" ]]; then
  require_path "${INIT_FROM}" "INIT_FROM must point at an existing checkpoint."
  MODEL_PATH="${INIT_FROM}"
fi

RUN_NAME="${RUN_NAME:-${STUDENT}_${DATASET}_sft}"
RUN_DIR="${SFT_DIR}/${RUN_NAME}"

require_path "${MODEL_PATH}" "Download the student checkpoint or set PRETRAINED_DIR."
require_path "${TRAIN_YAML}" "Missing training config."
require_path "${DEEPSPEED_CONFIG}" "Missing DeepSpeed config."
require_path "${DATASET_INFO}" "Build a dataset first (stages 3-5)."

if ! command -v llamafactory-cli >/dev/null 2>&1; then
  echo "[ERROR] llamafactory-cli not found. Install LLaMA-Factory; see the README." >&2
  exit 1
fi

banner "Stage 6: SFT (${STUDENT})"
echo "  Init from    : ${MODEL_PATH}"
echo "  Dataset      : ${DATASET}"
echo "  Dataset dir  : ${TRAIN_DATA_DIR}"
echo "  Config       : ${TRAIN_YAML}"
echo "  DeepSpeed    : ${DEEPSPEED_CONFIG}"
echo "  Output dir   : ${RUN_DIR}"

mkdir -p "${RUN_DIR}"

llamafactory-cli train "$(abspath "${TRAIN_YAML}")" \
  model_name_or_path="$(abspath "${MODEL_PATH}")" \
  deepspeed="$(abspath "${DEEPSPEED_CONFIG}")" \
  dataset_dir="$(abspath "${TRAIN_DATA_DIR}")" \
  dataset="${DATASET}" \
  output_dir="$(abspath "${RUN_DIR}")" \
  2>&1 | tee "${RUN_DIR}/train.log"

banner "Stage 6 complete. Checkpoint in ${RUN_DIR}"
