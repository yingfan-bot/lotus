#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-args/gsm8k_cot_llama1b.yaml}"
RUN_NAME="${RUN_NAME:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"

NAME_ARG=()
if [[ -n "$RUN_NAME" ]]; then
  NAME_ARG=(--name "$RUN_NAME")
fi

python -m torch.distributed.run \
  --nnodes 1 \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_port "$MASTER_PORT" \
  scripts/run.py "$CONFIG" "${NAME_ARG[@]}"
