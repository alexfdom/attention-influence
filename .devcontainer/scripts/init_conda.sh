#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${CONDA_ENV_NAME}"
ENV_FILE="$(pwd)/environment.yml"

echo "Creating / Updating Conda env '$ENV_NAME'…"
if conda info --envs | grep -q "$ENV_NAME"; then
  mamba env update -n "$ENV_NAME" -f "$ENV_FILE"
else
  mamba env create -n "$ENV_NAME" -f "$ENV_FILE"
fi

# quick sanity checks
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

python -m pip install --no-cache-dir --upgrade "pip==25.1.1"

python - << EOF
import torch, json, sys
info = {
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count()
}
print(json.dumps(info, indent=2))
sys.exit(0 if info['cuda_available'] else 1)
EOF
