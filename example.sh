#!/usr/bin/env bash

set -euo pipefail

python dreamerv3/main.py \
  --jax.platform cpu \
  --logdir ./logs/ \
  --configs debug \
  --run.steps 500
