# [PCS 2025] Exploring Autoregressive Vision Foundation Models for Image Compression

Our paper has been accepted to PCS 2025. This repository contains the source code and more visualizations for our paper.

## Abstract

This work presents the first attempt to repurpose vision foundation models (VFMs) as image codecs, aiming to explore their generation capability for low-rate image compression. VFMs are widely employed in both conditional and unconditional generation scenarios across diverse downstream tasks, e.g., physical AI applications. Many VFMs employ an encoder-decoder architecture similar to that of end-to-end learned image codecs and learn an autoregressive (AR) model to perform next-token prediction. To enable compression, we repurpose the AR model in VFM for entropy coding the next token based on previously coded tokens. This approach deviates from early semantic compression efforts that rely solely on conditional generation for reconstructing input images. Extensive experiments and analysis are conducted to compare VFM-based codec to current SOTA codecs optimized for distortion or perceptual quality. Notably, certain pre-trained, general-purpose VFMs demonstrate superior perceptual quality at extremely low bitrates compared to specialized learned image codecs. This finding paves the way for a promising research direction that leverages VFMs for low-rate, semantically rich image compression.

## Pretrained Models and Environment Setup

We evaluate four autoregressive vision foundation models: **VAR**, **LlamaGen**, **Cosmos**, and **Lumina-mGPT**. Please use the corresponding pretrained weights and environment for each model.

### VAR

**Repository:** [FoundationVision/VAR](https://github.com/FoundationVision/VAR)

**Pretrained weights:**
- VAR-d36 (2.3B, 512×512): [var_d36.pth](https://huggingface.co/FoundationVision/var/resolve/main/var_d36.pth)
- VQ-VAE: [vae_ch160v4096z32.pth](https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth)

**Environment:**

```bash
git clone https://github.com/FoundationVision/VAR.git
cd VAR

pip install torch>=2.0.0
pip install -r requirements.txt
```

---

### LlamaGen

**Repository:** [FoundationVision/LlamaGen](https://github.com/FoundationVision/LlamaGen)

**Pretrained weights:**
- LlamaGen-3B (3.1B, 384×384): [c2i_3B_384.pt](https://huggingface.co/FoundationVision/LlamaGen/resolve/main/c2i_3B_384.pt)
- VQ tokenizer: [vq_ds16_c2i.pt](https://huggingface.co/FoundationVision/LlamaGen/resolve/main/vq_ds16_c2i.pt)

**Environment:**

```bash
git clone https://github.com/FoundationVision/LlamaGen.git
cd LlamaGen

pip install -r requirements.txt
```

---

### Cosmos

**Repository:** [NVIDIA Cosmos](https://github.com/NVIDIA/Cosmos)

**Pretrained weight:**
- Cosmos-1.0-Autoregressive-12B: [Hugging Face](https://huggingface.co/nvidia/Cosmos-1.0-Autoregressive-12B)

**Environment:**

```bash
git clone https://github.com/NVIDIA/Cosmos.git
cd Cosmos

conda create -n cosmos python=3.10 -y
conda activate cosmos

pip install -r requirements.txt
```

Please follow the official Cosmos repository for downloading the corresponding visual tokenizer and additional model assets.

---

### Lumina-mGPT

**Repository:** [Alpha-VLLM/Lumina-mGPT](https://github.com/Alpha-VLLM/Lumina-mGPT)

**Pretrained weight:**
- Lumina-mGPT-7B: [Lumina-mGPT-7B-768](https://huggingface.co/Alpha-VLLM/Lumina-mGPT-7B-768)

**Environment:**

```bash
git clone https://github.com/Alpha-VLLM/Lumina-mGPT.git
cd Lumina-mGPT

conda create -n lumina_mgpt python=3.10 -y
conda activate lumina_mgpt

pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e .
```

Please follow the official Lumina-mGPT repository for downloading the required Chameleon VQ-VAE tokenizer weights.

## Visualizations

![Visualizations](images/comparison_pcs_kodim01.png)
![Visualizations](images/comparison_pcs_kodim02.png)
![Visualizations](images/comparison_pcs_kodim03.png)
![Visualizations](images/comparison_pcs_kodim04.png)
## Citation
If you find our project useful, please cite the following paper:
```
@inproceedings{VFMcodec,
    title     = {Exploring Autoregressive Vision Foundation Models for Image Compression},
  author={Huu-Tai Phung and Yu-Hsiang Lin and Yen-Kuan Ho and Wen-Hsiao Peng},

    booktitle = {Proceedings of Picture Coding Symposium (PCS)},
    year      = {2025}
}
```
