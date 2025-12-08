#!/bin/bash

SOURCE_FILE="/mnt/public/wph/codes/RLinf/examples/embodiment/config/env/eval/robotwin_single_task.yaml"

# 目标文件列表（元素间用空格分隔，无逗号）
TARGET_FILES=(
    "/mnt/public/wph/codes/RLinf/examples/embodiment/config/env/eval/robotwin_beat_block_hammer.yaml"
    "/mnt/public/wph/codes/RLinf/examples/embodiment/config/env/eval/robotwin_place_container_plate.yaml"
    "/mnt/public/wph/codes/RLinf/examples/embodiment/config/env/eval/robotwin_handover_block.yaml"
    "/mnt/public/wph/codes/RLinf/examples/embodiment/config/env/eval/robotwin_lift_pot.yaml"
    "/mnt/public/wph/codes/RLinf/examples/embodiment/config/env/eval/robotwin_stack_bowls_two.yaml"
    "/mnt/public/wph/codes/RLinf/examples/embodiment/config/env/eval/robotwin_move_can_pot.yaml"
)

# 循环复制
for target in "${TARGET_FILES[@]}"; do
    echo "复制到目标文件: $target"
    cp "$SOURCE_FILE" "$target"
    if [ $? -eq 0 ]; then
        echo "复制到 $target 成功"
    else
        echo "复制到 $target 失败"
    fi
done