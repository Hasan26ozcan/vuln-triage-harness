#!/bin/sh
# Entrypoint for the Stage 9 GPU serving image.
#
# Exists so the Dockerfile's CMD can use exec form (`CMD ["/app/entrypoint.sh"]`)
# instead of shell form. Shell-form CMD runs as a child of `/bin/sh -c "..."`,
# which does not forward SIGTERM to the child process by default -- `docker
# stop` then has to wait out the full grace period and SIGKILL the container.
# `exec` below replaces this shell process with python (same PID), so signals
# from the container runtime go straight to the server process.
set -eu

exec python3.11 -m app.serving.cli \
    --model-path "$MODEL_PATH" \
    --backend "$BACKEND_TYPE" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --num-ctx "$NUM_CTX" \
    --host "$HOST" \
    --port "$PORT"
