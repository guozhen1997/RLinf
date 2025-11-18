#! /bin/bash
set -x

tabs 4
CONFIG_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname $(dirname "$CONFIG_PATH"))
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

if [ -z "$1" ]; then
    CONFIG_NAME="sft"
else
    CONFIG_NAME=$1
fi

python ${REPO_PATH}/examples/sft/main_sft.py --config-path ${CONFIG_PATH}/config/  --config-name $CONFIG_NAME
