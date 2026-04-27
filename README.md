# Cross-Distribution Diffusion Priors-Driven Iterative Reconstruction for Sparse-View CT (CDPIR)

This repository provides the PyTorch implementation, pretrained weights, and training/sampling code for our **TMI** paper on **Cross-Distribution Diffusion Priors-Driven Iterative Reconstruction for Sparse-View CT (CDPIR)**.

> [**Cross-Distribution Diffusion Priors-Driven Iterative Reconstruction for Sparse-View CT**](https://arxiv.org/pdf/2509.13576)  
> [Haodong Li](https://graeme-lee.github.io/haodong/), [Hengyong Yu](https://hengyongyu.wixsite.com/ctlab/untitled-cyjn)  
> University of Massachusetts Lowell

<p align="center">
  <img src="figures/figure_architecture.JPG" width="800" alt="CDPIR Model Architecture">
</p>

<p align="center">
  <img src="figures/attention.png" width="800" alt="Attention Visualization">
</p>

## Overview

We propose **Cross-Distribution Diffusion Priors-Driven Iterative Reconstruction (CDPIR)** to address out-of-distribution (OOD) challenges in **sparse-view CT (SVCT)** reconstruction. CDPIR integrates **cross-distribution diffusion priors**, learned with a **Scalable Interpolant Transformer (SiT)** backbone, into a **model-based iterative reconstruction** framework.

By establishing a unified stochastic interpolant framework and leveraging **Classifier-Free Guidance (CFG)**, CDPIR learns a highly transferable prior that preserves domain-invariant anatomical structures while allowing domain-specific appearance modulation. Through alternating **data-fidelity updates** and **diffusion sampling steps**, CDPIR achieves strong robustness and excellent detail preservation, significantly outperforming existing methods in challenging OOD scenarios.

> This codebase is modified from [SiT](https://github.com/willisma/SiT) by N. Ma, S. Xie, et al.

---

## 🚀 Getting Started

### 1. Set up the CDPIR environment

We provide an `environment.yml` file for creating the Conda environment:

```bash
conda env create -f environment.yml
conda activate cdpir
```

2. **Setup the environment of LEAP:** Install the [LEAP](https://github.com/LLNL/LEAP) projector library. We use [manual](https://github.com/llnl/LEAP/blob/main/manual_install.py) installing based on the [pre-complied files](https://github.com/LLNL/LEAP/wiki/Using-the-LEAP-precompiled-dynamic-libraries). 
3. **Download assets:** Download the [pre-trained ckpt](https://huggingface.co/Hd9955/CDPIR/blob/main/0200000.pt).

> **Note:** In the provided pretrained checkpoint, the AAPM label is set to '''0''', and the COCA label is set to '''1'''.

## ⚙️ 2D Simulation Reconstruction

After preparing the pretrained weights and test data, you can run reconstruction using:

```bash
python sample.py SDE 
```
To change the experimental settings, modify the parameters directly in the Python scripts.

For raw-injection input, you can replace the simulated input projection with raw projection data in the code.

## ⚙️ CDPIR Training
The training is based on the [DiT](https://github.com/facebookresearch/dit) training. You may also use the ODE sampler during testing to monitor whether the model has been trained sufficiently.
```bash
python  train.py --model SiT-B/2 --data-path /path/to/imagenet/train
```
## 📑 Citation
If you find our paper helpful, please kindly cite our paper in your publications.
```bash
@misc{li2025crossdistributiondiffusionpriorsdriveniterative,
      title={Cross-Distribution Diffusion Priors-Driven Iterative Reconstruction for Sparse-View CT}, 
      author={Haodong Li and Shuo Han and Haiyang Mao and Yu Shi and Changsheng Fang and Jianjia Zhang and Weiwen Wu and Hengyong Yu},
      year={2025},
      eprint={2509.13576},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2509.13576}, 
}
```




