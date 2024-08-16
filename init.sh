#!/usr/bin/env bash

set -euo pipefail

python -m venv .venv

source .venv/bin/activate

pip install uv

uv pip install -r embodied/requirements.txt -r dreamerv3/requirements.txt
