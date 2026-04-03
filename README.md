# Cross-Distribution Diffusion Priors-Driven Iterative Reconstruction for Sparse View CT(CDPIR)

This repo contains PyTorch model definitions, pre-trained weights, and training/sampling code for our paper exploring Cross-Distribution Diffusion Priors-Driven Iterative Reconstruction for Sparse-View CT (CDPIR).

> [Cross-Distribution Diffusion Priors-Driven Iterative Reconstruction for Sparse-View CT](https://arxiv.org/pdf/2509.13576)
> [Haodong Li](https://graeme-lee.github.io/haodong/) , [Hengyong Yu](https://hengyongyu.wixsite.com/ctlab/untitled-cyjn)
> University of Massachusetts

![Model Architecture](figure_architecture.JPG)
![Figure1](attention.png)


We present Cross-Distribution Diffusion Priors-Driven Iterative Reconstruction (CDPIR) to tackle out-of-distribution (OOD) challenges in Sparse-View CT (SVCT). CDPIR integrates cross-distribution diffusion priors, derived from a Scalable Interpolant Transformer (SiT) backbone, with model-based iterative reconstruction. By establishing a unified stochastic interpolant framework and leveraging Classifier-Free Guidance (CFG), our model learns a highly transferable prior that preserves domain-invariant anatomical structures while allowing domain-specific appearance modulations. By alternating between data fidelity and sampling updates, CDPIR achieves state-of-the-art detail preservation and robustness, significantly outperforming existing approaches in challenging OOD scenarios.  This code is modified by [SiT(N Ma, S Xie.etl)](https://github.com/willisma/SiT). 



Get Started

1.Setup the environment of SiT

2.Setup the environment of LEAP

3.Download the pre-trained ckpts and the test XCAT CT slices and the GE projections

By default, the above scripts place the pre-trained model checkpoints under pretrained, and the sample data under data.

CDPIR reconstruction
Once you have the pre-trained weights and the test data set up properly, you may run the following scripts. Modify the parameters in the python scripts directly to change experimental settings.

```
conda activate cdpir
python sample.py RECON
```
