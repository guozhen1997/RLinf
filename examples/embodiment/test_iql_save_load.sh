#!/bin/bash
# Test IQL checkpoint save and resume.
# Phase 1: train 2 steps and save at step 2.
# Phase 2: resume from that checkpoint and run 2 more steps (to step 4).
set -e
export EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH="$(dirname "$(dirname "$EMBODIED_PATH")")"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

LOG_DIR="${REPO_PATH}/logs/iql_save_load_test"
RESUME_DIR="${LOG_DIR}/iql_save_test/checkpoints/global_step_2"

echo "[Phase 1] Training 2 steps (save at step 2)..."
python "${EMBODIED_PATH}/train_offlinerl.py" \
  --config-path "${EMBODIED_PATH}/config" \
  --config-name d4rl_offline_mujoco \
  runner.logger.log_path="${LOG_DIR}" \
  runner.logger.experiment_name=iql_save_test \
  runner.max_steps=2 \
  runner.save_interval=2 \
  runner.val_check_interval=2 \
  runner.local_update_steps=4

if [[ ! -d "${RESUME_DIR}/actor" ]]; then
  echo "Expected checkpoint not found: ${RESUME_DIR}/actor"
  exit 1
fi
echo "[Phase 1] Checkpoint saved at ${RESUME_DIR}"

echo "[Phase 2] Resuming from checkpoint and running to step 4..."
python "${EMBODIED_PATH}/train_offlinerl.py" \
  --config-path "${EMBODIED_PATH}/config" \
  --config-name d4rl_offline_mujoco \
  runner.logger.log_path="${LOG_DIR}" \
  runner.logger.experiment_name=iql_save_test_resume \
  runner.resume_dir="${RESUME_DIR}" \
  runner.max_steps=4 \
  runner.save_interval=4 \
  runner.val_check_interval=4 \
  runner.local_update_steps=4

echo "[OK] IQL save and load test passed."
