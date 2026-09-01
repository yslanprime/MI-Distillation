#!/usr/bin/env bash
# Stage 2: sample the candidate pool D (Eq. 6) from every interpolated teacher.
#
# Writes one JSON per lambda. Resumable: rerunning skips problems already
# present in the output file.
#
# Usage:
#   bash scripts/02_generate_cot.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 LAMBDAS="0.4" bash scripts/02_generate_cot.sh

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

require_path "${TRAIN_PROBLEMS}" "Set TRAIN_PROBLEMS to the MATH training split (parquet file or directory)."

banner "Stage 2: generating CoT trajectories"
echo "  Problems     : ${TRAIN_PROBLEMS}"
echo "  Teachers     : ${TEACHER_DIR}/${TEACHER_TAG}_lambda*"
echo "  Lambdas      : ${LAMBDAS}"
echo "  Output dir   : ${COT_DIR}/${DATASET_NAME}_${TEACHER_TAG}"
echo "  Sampling     : T=${TEMPERATURE}, top-p=${TOP_P}, max_new_tokens=${MAX_NEW_TOKENS}"
echo "  TP size      : ${TENSOR_PARALLEL_SIZE}"

failed=()

for lam in ${LAMBDAS}; do
  lam_tag="$(lambda_tag "${lam}")"
  teacher_path="$(teacher_path_for_lambda "${lam}")"
  output_file="$(cot_file_for_lambda "${lam}")"

  if [[ ! -e "${teacher_path}" ]]; then
    echo ""
    echo "[skip] Teacher not found for lambda=${lam_tag}: ${teacher_path}"
    echo "       Run scripts/01_interpolate_teachers.sh first."
    failed+=("lambda=${lam_tag} (missing teacher)")
    continue
  fi

  echo ""
  echo "------------------------------------------------------------"
  echo " lambda = ${lam_tag}"
  echo "   Teacher : ${teacher_path}"
  echo "   Output  : ${output_file}"
  echo "------------------------------------------------------------"

  if "${PYTHON_BIN}" src/generate_cot.py \
      --model_path           "${teacher_path}" \
      --data_path            "${TRAIN_PROBLEMS}" \
      --output_file          "${output_file}" \
      --instruction          "${EVAL_INSTRUCTION}" \
      --max_new_tokens       "${MAX_NEW_TOKENS}" \
      --temperature          "${TEMPERATURE}" \
      --top_p                "${TOP_P}" \
      --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
      --save_every           "${SAVE_EVERY}"; then
    echo "[done] lambda=${lam_tag} -> ${output_file}"
  else
    echo "[fail] lambda=${lam_tag}; continuing with the remaining teachers." >&2
    failed+=("lambda=${lam_tag} (generation error)")
  fi
done

echo ""
if ((${#failed[@]} > 0)); then
  banner "Stage 2 finished with ${#failed[@]} problem(s)"
  for item in "${failed[@]}"; do
    echo "  - ${item}"
  done
  exit 1
fi

banner "Stage 2 complete. Candidate pool in ${COT_DIR}/${DATASET_NAME}_${TEACHER_TAG}"
