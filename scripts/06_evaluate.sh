#!/usr/bin/env bash
# Stage 6: evaluate distilled students on the reasoning benchmarks.
#
# Repeats each benchmark 16 times (AIME24, AMC23) or 4 times (the rest) and
# writes one JSON per (benchmark, model, run).
#
# Usage:
#   MODELS="outputs/sft/qwen3b_seqlss_alpha4" bash scripts/06_evaluate.sh
#   MODELS="run_a run_b" DATASETS="math500 gsm8k" bash scripts/06_evaluate.sh

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Space-separated list of model directories to evaluate.
MODELS="${MODELS:-}"
# Benchmarks to run; defaults to the seven reported in Table 2.
DATASETS="${DATASETS:-aime24 amc23 gpqa_diamond gsm8k math500 minervamath olympiadbench}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-8192}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-16384}"
EVAL_TP_SIZE="${EVAL_TP_SIZE:-${TENSOR_PARALLEL_SIZE}}"
# Small test sets need more repeats to give a stable pass@1 estimate.
SMALL_BENCHMARKS="${SMALL_BENCHMARKS:-aime24 aime25 amc23}"
SMALL_RUNS="${SMALL_RUNS:-16}"
LARGE_RUNS="${LARGE_RUNS:-4}"

if [[ -z "${MODELS}" ]]; then
  echo "[ERROR] MODELS is required: a space-separated list of checkpoint directories." >&2
  echo "        e.g. MODELS=\"${SFT_DIR}/qwen3b_seqlss_alpha4_train7500_sft\"" >&2
  exit 1
fi

runs_for_dataset() {
  local dataset="$1"
  for small in ${SMALL_BENCHMARKS}; do
    if [[ "${dataset}" == "${small}" ]]; then
      echo "${SMALL_RUNS}"
      return
    fi
  done
  echo "${LARGE_RUNS}"
}

banner "Stage 7: evaluation"
echo "  Models     : ${MODELS}"
echo "  Benchmarks : ${DATASETS}"
echo "  Results    : ${EVAL_DIR}"
echo "  Decoding   : T=${TEMPERATURE}, top-p=${TOP_P}, max_new_tokens=${EVAL_MAX_NEW_TOKENS}"
echo "  Repeats    : ${SMALL_RUNS} for (${SMALL_BENCHMARKS}), ${LARGE_RUNS} otherwise"

mkdir -p "${EVAL_DIR}"

for dataset in ${DATASETS}; do
  num_runs="$(runs_for_dataset "${dataset}")"

  for model_path in ${MODELS}; do
    if [[ ! -d "${model_path}" ]]; then
      echo ""
      echo "[skip] Model directory not found: ${model_path}"
      continue
    fi
    model_name="$(basename "${model_path%/}")"

    for run in $(seq 1 "${num_runs}"); do
      output_file="${EVAL_DIR}/${dataset}_${model_name}_run${run}.json"
      if [[ -f "${output_file}" ]]; then
        echo "[skip] ${output_file} already exists."
        continue
      fi

      echo ""
      echo "------------------------------------------------------------"
      echo " ${dataset} | ${model_name} | run ${run}/${num_runs}"
      echo "------------------------------------------------------------"

      # A per-run seed keeps repeats independent but reproducible.
      "${PYTHON_BIN}" evaluation/evaluate.py \
        --dataset              "${dataset}" \
        --model_path           "${model_path}" \
        --output_file          "${output_file}" \
        --max_new_tokens       "${EVAL_MAX_NEW_TOKENS}" \
        --max_model_len        "${EVAL_MAX_MODEL_LEN}" \
        --temperature          "${TEMPERATURE}" \
        --top_p                "${TOP_P}" \
        --tensor_parallel_size "${EVAL_TP_SIZE}" \
        --seed                 "$((SEED + run))" \
        "$@"
    done
  done
done

banner "Stage 6 complete. Per-run results in ${EVAL_DIR}"
