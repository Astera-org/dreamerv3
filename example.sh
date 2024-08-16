#!/usr/bin/env bash

set -euo pipefail

source .venv/bin/activate

python dreamerv3/main.py \
  --jax.platform cpu \
  --logdir ./logs/ \
  --configs debug \
  --run.steps 500
