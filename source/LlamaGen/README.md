# How to run compression 

## Requirements

Place these model files in `LlamaGen/pretrained_models/`:

```text
pretrained_models/
├── vq_ds16_c2i.pt
└── c2i_3B_384.pt
```

A CUDA-capable GPU is required.

## Run Compression

```bash
bash /home/at9529/mick20001108.cs12/Tai/LlamaGen/compress_script.sh \
  /absolute/path/to/input.png
```

Example:

```bash
bash /home/at9529/mick20001108.cs12/Tai/LlamaGen/compress_script.sh \
  /home/at9529/mick20001108.cs12/Tai/cosmos-predict1/datasets/image_kodak/kodim01.png
```

## Output

Results are saved to:

```text
/home/at9529/mick20001108.cs12/Tai/LlamaGen/outputs/kodak_runtime/
```

To use a different output directory:

```bash
bash /home/at9529/mick20001108.cs12/Tai/LlamaGen/compress_script.sh \
  /absolute/path/to/input.png \
  --save-path ./outputs/my_run
```
