#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 1 ]]; then
    echo "Usage: bash $0 IMAGE_PATH [additional compression options]" >&2
    exit 2
fi
image_path="$(realpath -- "$1")"
shift
if [[ ! -f "$image_path" ]]; then
    echo "Input image not found: $image_path" >&2
    exit 2
fi
cd -- "$script_dir"
exec python3 autoregressive/sample/compress.py \
    --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt \
    --gpt-ckpt ./pretrained_models/c2i_3B_384.pt \
    --gpt-model GPT-3B --image-size 512 --vq-model VQ-16 --from-fsdp \
    --image-path "$image_path" --save-path ./outputs/kodak_runtime "$@"
