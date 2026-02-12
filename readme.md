# Install

### pip
```bash
conda create -n ctnexus python=3.12
conda activate ctnexus
pip install -e .
```

### uv (Recommended)
[uv](https://docs.astral.sh/uv/) is a fast Python package installer and resolver.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment with Python 3.12
uv venv --python 3.12

# Activate the virtual environment
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate  # On Windows

# Install the package in editable mode
uv pip install -e .
```

## Docker Image Building
```bash
# From CT-NEXUS root
cd CT-NEXUS
docker build -f Dockerfile -t ctnexus .
docker save ctnexus:latest | gzip > ctnexus.tar.gz
```

### Run Container
```bash
docker run --rm --gpus all -m 8G \
    -v ./inputs:/workspace/inputs \
    -v ./outputs:/workspace/outputs \
    ctnexus:latest /bin/bash -c "sh extract_feat.sh"
``` 

## Usage

Pre-process and format the data as you would for OpenMind as usual. Train the models as below.


```bash
# ResEnc-L
nnssl_train 745 onemmiso -tr AlignedMAEFTTrainer -p nnsslPlans -num_gpus 1 -pretrained_weights ${nnssl_results}/Dataset745_OpenMind/MAETrainer/fold_all/checkpoint_final.pth || true

#Primus-M
nnssl_train 745 onemmiso -tr AlignedMAEFTLR3EvaTrainer -p nnsslPlans -num_gpus 1 -pretrained_weights ${nnssl_results}/Dataset745_OpenMind/MAETrainer/fold_all/checkpoint_final.pth || true
```

----

## 📦 Pretrained Weights

Pretrained weights will be released soon on **Zenodo**. Stay tuned!

---

## 📖 Citation

Citation to our challenge report and paper.

Please also cite the original work this repo builds on:

```bibtex
@article{vaishConsistentViewAlignment2025,
  title = {Consistent View Alignment Improves Foundation Models for {{3D}} Medical Image Segmentation},
  author = {Vaish, Puru and Meister, Felix and Heimann, Tobias and Brune, Christoph and Wolterink, Jelmer M.},
  year = {2025},
  month = sep,
  number = {arXiv:2509.13846},
  eprint = {2509.13846},
  primaryclass = {cs},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2509.13846},
  url = {http://arxiv.org/abs/2509.13846},
  urldate = {2025-09-19},
  abstract = {Many recent approaches in representation learning implicitly assume that uncorrelated views of a data point are sufficient to learn meaningful representations for various downstream tasks. In this work, we challenge this assumption and demonstrate that meaningful structure in the latent space does not emerge naturally. Instead, it must be explicitly induced. We propose a method that aligns representations from different views of the data to align complementary information without inducing false positives. Our experiments show that our proposed self-supervised learning method, Consistent View Alignment, improves performance for downstream tasks, highlighting the critical role of structured view alignment in learning effective representations. Our method achieved first and second place in the MICCAI 2025 SSL3D challenge when using a Primus vision transformer and ResEnc convolutional neural network, respectively. The code and pretrained model weights are released at https://github.com/Tenbatsu24/LatentCampus.},
  archiveprefix = {arXiv},
  langid = {english},
  keywords = {Computer Science - Computer Vision and Pattern Recognition,Computer Science - Machine Learning},
}
```

---

## ⚖ License

This repository is released under the [Creative Commons Attribution-NonCommercial 4.0 International Public License](./LICENSE.md).

### Requirements

All requirements are the same as in the original repository, including dependencies for PyTorch, einops, thop, and other libraries.


