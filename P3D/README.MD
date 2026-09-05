# P3D: Highly Scalable 3D Neural Surrogates for Physics Simulations

<div align="center">

<p align="center">
<a href="https://pypi.org/project/p3d-surrogate/">
  <img src="https://img.shields.io/pypi/v/p3d-surrogate.svg" alt="PyPI">
</a> <a href="https://github.com/tum-pbs/p3d/blob/main/LICENSE.txt">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License">
</a>
</p>

[Paper](https://openreview.net/forum?id=8UdCE5nhFl) •
[Project Page](https://akanota.github.io/p3d/) •
[🤗 Hugging Face](https://huggingface.co/collections/thuerey-group/p3d)

</div>

---

**Authors:** [Benjamin Holzschuh](), [Georg Kohl](), [Florian Redinger](), [Nils Thuerey](https://ge.in.tum.de/about/n-thuerey/)

**P3D** is a scalable framework for learning neural surrogates for high-resolution 3D physics simulations. The P3D model series combines convolutional blocks for fast local feature processing with windowed self-attention for deep representation learning in a hierarchical U-shaped architecture. 

![Image](https://github.com/user-attachments/assets/4f50eae5-df66-46a3-92fb-f7c2ec92ed98)

## Quick Installation

```bash
# Install from PyPI
pip install p3d-surrogate

# Or install from source
git clone https://github.com/tum-pbs/P3D.git
cd P3D
pip install -e .
```

---

## Usage

P3D is designed to efficiently process and predict the evolution of physical systems described by partial differential equations (PDEs) on 3D volumetric grids.

```python
from p3d_surrogate import P3D_S
import torch

# Load pre-trained model
model = P3D_S(channel_size=4, channel_size_out=4).cuda()

# Run a forward pass on a 3D physics field
# Shape: (batch, channels, depth, height, width)
x_128 = torch.randn((1, 4, 128, 128, 128), dtype=torch.float32).cuda()
x_64 = torch.randn((1, 4, 64, 64, 64), dtype=torch.float32).cuda()

prediction_128 = model(x_128).sample
prediction_64 = model(x_64).sample
```

---

## Examples 

- [Isotropic Turbulence Rollout](notebooks/inference_isotropic.ipynb): Autoregressive rollout of isotropic turbulence.
- [Probabilistic Channel Flow Generation](notebooks/tcf_pretrained.ipynb): Generate turbulent channel flow using P3D-L and visualize the results.
- More to come!
---

## Datasets

- *APE3D*: Modified version of [APEBench](https://tum-pbs.github.io/apebench-paper/). For dataset generation, see `data/simulations_apebench`.
- *Isotropic Turbulence*: Available on [JHTDB](https://turbulence.idies.jhu.edu/datasets/homogeneousTurbulence/isotropic). See `data/download_jhtdb` for download instructions. We provide a single snapshot at `t=420` on [Hugging Face](https://huggingface.co/datasets/thuerey-group/jhtdb-isotropic-turbulence-1024).
- *Turbulence Channel Flow*: Data generation based on [PICT](https://github.com/tum-pbs/PICT). Data is provided on [Hugging Face](https://huggingface.co/datasets/thuerey-group/turbulent-channel-flow-pict).
## Citation

```bibtex
@inproceedings{holzschuh2026p3d,
  title     = {P3D: Highly Scalable 3D Neural Surrogates for Physics
               Simulations with Global Context},
  author    = {Benjamin Holzschuh and Georg Kohl and
               Florian Redinger and Nils Thuerey},
  booktitle = {The Fourteenth International Conference on
               Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=8UdCE5nhFl}
}
```