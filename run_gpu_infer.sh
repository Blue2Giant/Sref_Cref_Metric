#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /home/i-lanjinghong/miniconda3/etc/profile.d/conda.sh
conda activate Sref

exec python "${SCRIPT_DIR}/gpu_infer_ours.py" "$@"
