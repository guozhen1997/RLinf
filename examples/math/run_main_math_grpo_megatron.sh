#! /bin/bash
set -x

tabs 4
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0

CONFIG_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname $(dirname "$CONFIG_PATH"))
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

if [ -z "$1" ]; then
    CONFIG_NAME="qwen3-30b-grpo-megatron"
    # CONFIG_NAME="qwen2.5-1.5b-grpo-megatron"
else
    CONFIG_NAME=$1
fi

export PYTHONPATH=/mnt/public/rl_fuxu/Megatron-LM:$PYTHONPATH

python ${REPO_PATH}/examples/math/main_math.py --config-path ${CONFIG_PATH}/config/  --config-name $CONFIG_NAME