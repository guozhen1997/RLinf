#! /bin/bash
source /mnt/mnt/public/chenkang/zqlenv_wmj_0729/bin/activate

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

# NOTE: set LIBERO_REPO_PATH to the path of the LIBERO repo
export LIBERO_REPO_PATH="/opt/libero"

export PYTHONPATH=${REPO_PATH}:${LIBERO_REPO_PATH}:$PYTHONPATH

# Base path to the BEHAVIOR dataset, which is the BEHAVIOR-1k repo's dataset folder
# Only required when running the behavior experiment.
export OMNIGIBSON_DATA_PATH=$OMNIGIBSON_DATA_PATH
export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-$OMNIGIBSON_DATA_PATH/behavior-1k-assets/}
export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson.key}
export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_ASSET_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
# Base path to Isaac Sim, only required when running the behavior experiment.
export ISAAC_PATH=${ISAAC_PATH:-/path/to/isaac-sim}
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}


if [ -z "$1" ]; then
    CONFIG_NAME="maniskill_ppo_openpi"
else
    CONFIG_NAME=$1
fi


# Path to openpi repository
export PYTHONPATH=/mnt/mnt/public/zhangtonghe/openpi-main/src:$PYTHONPATH
export PYTHONPATH=/mnt/mnt/public/zhangtonghe/openpi-main/packages/openpi-client/src:$PYTHONPATH 
# ManiSkill asset dir path (change that to your own path)
export MANISKILL_ASSET_DIR=${MANISKILL_ASSET_DIR:-"${REPO_PATH}/rlinf/envs/maniskill/assets"}


MODEL_DIR="/mnt/mnt/public/zhangtonghe/openpi-main/checkpoints/sft/pi0_maniskill/PutOnPlateInScene25Main-v3/2025-10-29-22-52-21/250" 

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR} \
 rollout.model.model_path=${MODEL_DIR} \
 actor.model.model_path=${MODEL_DIR} \
 env.eval.video_cfg.save_video=False "
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}