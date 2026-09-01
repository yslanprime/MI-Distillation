#!/usr/bin/env bash
# Stage 1: build the Instruct-Reasoning teacher spectrum (Eq. 3).
#
# Endpoints (lambda 0.0 and 1.0) are symlinked rather than copied; set
# MATERIALIZE_ENDPOINTS=1 to write them out. Runs on CPU and needs ~130 GB RAM
# for two 32B bf16 checkpoints, plus ~65 GB disk per interpolated teacher.
#
# Usage:
#   bash scripts/01_interpolate_teachers.sh
#   LAMBDAS="0.4 0.6" bash scripts/01_interpolate_teachers.sh

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

INTERPOLATION_DEVICE="${INTERPOLATION_DEVICE:-cpu}"
INTERPOLATION_DTYPE="${INTERPOLATION_DTYPE:-bfloat16}"
MATERIALIZE_ENDPOINTS="${MATERIALIZE_ENDPOINTS:-0}"

require_path "${REASONING_MODEL}" "Set REASONING_MODEL or download the reasoning teacher first."
require_path "${INSTRUCT_MODEL}" "Set INSTRUCT_MODEL or download the instruct teacher first."

banner "Stage 1: interpolating teachers (${TEACHER_TAG})"
echo "  Reasoning endpoint : ${REASONING_MODEL}   (lambda = 0.0)"
echo "  Instruct endpoint  : ${INSTRUCT_MODEL}   (lambda = 1.0)"
echo "  Lambdas            : ${LAMBDAS}"
echo "  Output root        : ${TEACHER_DIR}"
echo "  Device / dtype     : ${INTERPOLATION_DEVICE} / ${INTERPOLATION_DTYPE}"

mkdir -p "${TEACHER_DIR}"

for lam in ${LAMBDAS}; do
  lam_tag="$(lambda_tag "${lam}")"
  output_path="$(teacher_path_for_lambda "${lam}")"

  if [[ -e "${output_path}" ]]; then
    echo ""
    echo "[skip] ${output_path} already exists."
    continue
  fi

  # The endpoints need no interpolation; link them so downstream stages can
  # address every lambda uniformly.
  if [[ "${MATERIALIZE_ENDPOINTS}" != "1" ]] && { [[ "${lam_tag}" == "0.0" ]] || [[ "${lam_tag}" == "1.0" ]]; }; then
    if [[ "${lam_tag}" == "0.0" ]]; then
      endpoint="${REASONING_MODEL}"
    else
      endpoint="${INSTRUCT_MODEL}"
    fi
    echo ""
    echo "[link] lambda=${lam_tag} is an endpoint -> ${endpoint}"
    ln -s "$(cd "${endpoint}" && pwd)" "${output_path}"
    continue
  fi

  echo ""
  echo "------------------------------------------------------------"
  echo " lambda = ${lam_tag}  ->  ${output_path}"
  echo "------------------------------------------------------------"
  "${PYTHON_BIN}" src/interpolate_models.py \
    --reasoning_model_path "${REASONING_MODEL}" \
    --instruct_model_path  "${INSTRUCT_MODEL}" \
    --output_path          "${output_path}" \
    --lam                  "${lam}" \
    --dtype                "${INTERPOLATION_DTYPE}" \
    --device               "${INTERPOLATION_DEVICE}"
done

banner "Stage 1 complete. Teachers under ${TEACHER_DIR}"
