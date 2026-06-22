#!/bin/bash

# Run STEAM value model SFT training
# Usage: bash examples/value/steam/run_steam_sft.sh [CONFIG_NAME] [EXTRA_ARGS...]
# Example: bash examples/value/steam/run_steam_sft.sh steam_model_ensemble1
# Example: bash examples/value/steam/run_steam_sft.sh steam_model_ensemble1 data.k=8
# Example: bash examples/value/steam/run_steam_sft.sh steam_model_ensemble1 data.tag=my_tag

export SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
export REPO_PATH="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export EMBODIED_PATH="${SCRIPT_DIR}"
export SRC_FILE="${SCRIPT_DIR}/train_steam.py"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HOME}/.cache/transformers}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# Suppress libdav1d/ffmpeg verbose logging
export AV_LOG_FORCE_NOCOLOR=1
export LIBAV_LOG_LEVEL=quiet
export OPENCV_LOG_LEVEL=off
export FFREPORT=""

export PYTHONPATH="${REPO_PATH}:$PYTHONPATH"

# Reduces allocator fragmentation under FSDP — important when peak
# allocations (e.g. per-FSDP-unit gathered bf16 forward copy) compete with
# already-allocated fp32 master params + optimizer state.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

source switch_env openpi 2>/dev/null || echo "Warning: switch_env not found, using current environment"

if [ -z "$1" ]; then
    CONFIG_NAME="steam_model_ensemble1"
else
    CONFIG_NAME=$1
fi
shift 1 2>/dev/null || true
EXTRA_ARGS="$@"

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/steam_sft/${CONFIG_NAME}-$(date +'%Y%m%d-%H:%M:%S')"
LOG_FILE="${LOG_DIR}/run_steam_sft.log"
mkdir -p "${LOG_DIR}"
# Resolve shared config groups (training_backend/...) from examples/sft/config
# regardless of what searchpath an individual config file declares.
HYDRA_ARGS=(
    "runner.logger.log_path=${LOG_DIR}"
    "hydra.searchpath=[file://${SCRIPT_DIR}/config/,file://${REPO_PATH}/examples/sft/config/]"
)
CMD_BASE="python ${SRC_FILE} --config-path ${SCRIPT_DIR}/config/ --config-name ${CONFIG_NAME}"
echo "${CMD_BASE} ${HYDRA_ARGS[*]} ${EXTRA_ARGS}" > "${LOG_FILE}"
${CMD_BASE} "${HYDRA_ARGS[@]}" ${EXTRA_ARGS} 2>&1 | grep -v "libdav1d" | tee -a "${LOG_FILE}"
# The grep|tee pipeline would otherwise return tee's exit status and mask a
# training failure; propagate the Python process's exit code instead.
exit "${PIPESTATUS[0]}"
