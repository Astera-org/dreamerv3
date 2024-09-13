#!/usr/bin/env bash

set -euo pipefail

python dreamerv3/main.py --logdir 'logs/{timestamp}' --configs minetest --task minetest_boad

# or python dreamerv3/main.py --logdir logs/{timestamp} --configs minetest size12m --run.eval_every 120 --batch_size 1 --task minetest_boad
