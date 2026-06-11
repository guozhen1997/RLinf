#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_vla_sft.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

# Suppress libdav1d / ffmpeg verbose logging.
export AV_LOG_FORCE_NOCOLOR=1
export LIBAV_LOG_LEVEL=quiet
export FFREPORT=""

export PYTHONPATH=${REPO_PATH}:${LIBERO_REPO_PATH}:$PYTHONPATH

export DREAMZERO_PATH=${DREAMZERO_PATH:-"/path/to/DreamZero"}
export PYTHONPATH=${DREAMZERO_PATH}:$PYTHONPATH

if [ -z "$1" ]; then
    CONFIG_NAME="maniskill_ppo_openvlaoft"
else
    CONFIG_NAME=$1
fi
shift 1 2>/dev/null || true
EXTRA_ARGS=("$@")

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/sft/${CONFIG_NAME}-$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
HYDRA_ARGS=("runner.logger.log_path=${LOG_DIR}")
CMD_BASE=(python "${SRC_FILE}" --config-path "${EMBODIED_PATH}/config/" --config-name "${CONFIG_NAME}")
echo "${CMD_BASE[*]} ${HYDRA_ARGS[*]} ${EXTRA_ARGS[*]}" > ${MEGA_LOG_FILE}
"${CMD_BASE[@]}" "${HYDRA_ARGS[@]}" "${EXTRA_ARGS[@]}" 2>&1 | grep -v "libdav1d" | tee -a ${MEGA_LOG_FILE}
