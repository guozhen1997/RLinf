#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
export REPO_PATH=$(dirname "$(dirname "$EMBODIED_PATH")")
export SRC_FILE="${EMBODIED_PATH}/train_offlinerl.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"

if [ -z "$1" ]; then
    CONFIG_NAME="d4rl_offline_mujoco"
else
    CONFIG_NAME=$1
fi
shift 1 2>/dev/null || true
EXTRA_ARGS=("$@")

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
MEGA_LOG_FILE="${LOG_DIR}/run_offlinerl.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR} runner.logger.experiment_name=${CONFIG_NAME}"

# Append hydra overrides; when actor.use_fsdp_wrap=true, auto-add FSDP defaults unless overridden or meta
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    CMD="${CMD} ${EXTRA_ARGS[*]}"
    ADD_FSDP_DEFAULTS=false
    for arg in "${EXTRA_ARGS[@]}"; do
        case "${arg}" in
            --cfg|--resolve|--help|--hydra-help|--info|--multirun|--run|--version)
                ADD_FSDP_DEFAULTS=false
                break
                ;;
            actor.use_fsdp_wrap=true|+actor.use_fsdp_wrap=true)
                ADD_FSDP_DEFAULTS=true
                ;;
            actor.fsdp_config.use_orig_params=*|+actor.fsdp_config.use_orig_params=*|actor.fsdp_config.sharding_strategy=*|+actor.fsdp_config.sharding_strategy=*)
                ADD_FSDP_DEFAULTS=false
                break
                ;;
        esac
    done
    [ "${ADD_FSDP_DEFAULTS}" = true ] && CMD="${CMD} actor.fsdp_config.use_orig_params=true actor.fsdp_config.sharding_strategy=no_shard"
fi

echo "${CMD}" > "${MEGA_LOG_FILE}"
${CMD} 2>&1 | tee -a "${MEGA_LOG_FILE}"
