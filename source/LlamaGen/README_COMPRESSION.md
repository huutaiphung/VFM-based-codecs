# LlamaGen compression code

Minimal source bundle for `compress_script.sh`, preserving local GPT, generation,
and VQ model modifications. See LICENSE for the upstream license.

## Run

Use Python 3.10 or newer and a CUDA GPU with enough memory for GPT-3B.
Install compatible CUDA builds of PyTorch and torchvision, then:

```bash
python3 -m pip install -r requirements-compress.txt
bash compress_script.sh /absolute/path/to/input.png
```

Supply these checkpoints separately in `pretrained_models/`:

- `vq_ds16_c2i.pt`
- `c2i_3B_384.pt`

The default model configuration uses `--image-size 512`. Supply inputs with
compatible token counts; this script does not resize images. Additional CLI
options may follow the image path, for example `--save-path ./outputs/my_run`.

Results are reconstructed images and estimated rates/timing in `rate.csv`.
The current implementation does not write an entropy-coded bitstream.

## Package for Git

Run `python3 package_compression.py` to create
`../LlamaGen_compression_code.zip` from an explicit file allowlist.
Extract the ZIP into a fresh repository to publish only the compression code.
Outputs, checkpoints, datasets, caches, Git history, notebooks, demos, training
code, and the separate PerceptualSimilarity checkout are excluded.
Original files are retained in the working directory.

The original repository already tracks a VGG checkpoint; `.gitignore` does not
remove tracked files or historical blobs. The ZIP includes no Git history.

Syntax and CLI checks do not establish inference correctness. Full inference
requires CUDA, the checkpoints, and an input image.
