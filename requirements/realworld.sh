export UV_PATH="."

uv venv franka && source ${UV_PATH}/franka/bin/activate && \
UV_TORCH_BACKEND=auto uv sync --active && \
uv sync --extra franka --active