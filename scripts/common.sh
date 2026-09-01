#!/usr/bin/env bash
# Shared defaults for the MI-Distillation pipeline scripts.
# Sourced, not executed. Every variable can be overridden from the environment.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"

# --- Layout -----------------------------------------------------------------
PRETRAINED_DIR="${PRETRAINED_DIR:-pretrained_model}"   # downloaded checkpoints
TEACHER_DIR="${TEACHER_DIR:-teachers}"                 # interpolated teachers
DATA_DIR="${DATA_DIR:-data}"
RAW_DATA_DIR="${RAW_DATA_DIR:-${DATA_DIR}/raw}"
COT_DIR="${COT_DIR:-${DATA_DIR}/cot_generated}"        # candidate pool D
FILTER_DIR="${FILTER_DIR:-${DATA_DIR}/filter_data}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-${DATA_DIR}/train_data}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
SFT_DIR="${SFT_DIR:-${OUTPUT_DIR}/sft}"
EVAL_DIR="${EVAL_DIR:-${OUTPUT_DIR}/eval}"

# Datasets are registered here by stages 3 and 4, then referenced by name from
# llamafactory-cli in stage 5.
DATASET_INFO="${DATASET_INFO:-${TRAIN_DATA_DIR}/dataset_info.json}"

# --- Teachers ---------------------------------------------------------------
# 32B spectrum, used for every result in the main paper.
REASONING_MODEL="${REASONING_MODEL:-${PRETRAINED_DIR}/QwQ-32B}"
INSTRUCT_MODEL="${INSTRUCT_MODEL:-${PRETRAINED_DIR}/Qwen2.5-32B-Instruct}"
TEACHER_TAG="${TEACHER_TAG:-Qwen2.5-32B}"

# --- Instruct-Reasoning spectrum -------------------------------------------
# lambda weights the Instruct endpoint: 0.0 is Long CoT, 1.0 is Short CoT.
LAMBDAS="${LAMBDAS:-0.0 0.2 0.4 0.6 0.8 1.0}"

# --- Data -------------------------------------------------------------------
DATASET_NAME="${DATASET_NAME:-math_train}"
TRAIN_PROBLEMS="${TRAIN_PROBLEMS:-${RAW_DATA_DIR}/${DATASET_NAME}}"
TRAIN_SIZE="${TRAIN_SIZE:-7500}"
VAL_SIZE="${VAL_SIZE:-500}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
SEED="${SEED:-42}"

# --- Generation / sampling (Guo et al., 2025) ------------------------------
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
SAVE_EVERY="${SAVE_EVERY:-500}"

# --- SeqLSS -----------------------------------------------------------------
ALPHA="${ALPHA:-4}"                 # learnability penalty; paper default
TOKENIZER_MODEL="${TOKENIZER_MODEL:-${PRETRAINED_DIR}/Qwen2.5-3B-Instruct}"

# Kept in its own single-quoted variable: the braces in \boxed{} would
# terminate a ${VAR:-default} expansion early and corrupt the prompt.
DEFAULT_EVAL_INSTRUCTION='Please reason step by step, and put your final answer within \boxed{}.'
EVAL_INSTRUCTION="${EVAL_INSTRUCTION-${DEFAULT_EVAL_INSTRUCTION}}"

# --- Helpers ----------------------------------------------------------------

# lambda_tag 0.4 -> "0.4"; used to build consistent file and directory names.
lambda_tag() {
  awk -v value="$1" 'BEGIN { printf "%.1f", value }'
}

teacher_path_for_lambda() {
  local lam
  lam="$(lambda_tag "$1")"
  echo "${TEACHER_DIR}/${TEACHER_TAG}_lambda${lam}"
}

cot_file_for_lambda() {
  local lam
  lam="$(lambda_tag "$1")"
  echo "${COT_DIR}/${DATASET_NAME}_${TEACHER_TAG}/${DATASET_NAME}_${TEACHER_TAG}_lambda${lam}.json"
}

# Resolve a path to absolute form, leaving already-absolute paths untouched.
abspath() {
  case "$1" in
    /*) echo "$1" ;;
    *)  echo "${ROOT_DIR}/$1" ;;
  esac
}

require_path() {
  if [[ ! -e "$1" ]]; then
    echo "[ERROR] Missing required path: $1" >&2
    echo "        $2" >&2
    exit 1
  fi
}

banner() {
  echo "============================================================"
  echo " $*"
  echo "============================================================"
}
