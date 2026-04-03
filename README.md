# Cross-Distribution Diffusion Priors-Driven Iterative Reconstruction for Sparse View CT(CDPIR)
Official PyTorch implementation of CDPIR: A Cross-Distribution Diffusion Priors-Driven Iterative Reconstruction framework using SiT to tackle out-of-distribution (OOD) challenges and achieve SOTA performance in Sparse-View CT.

[Cross-Distribution Diffusion Priors-Driven Iterative Reconstruction for Sparse-View CT](https://arxiv.org/pdf/2509.13576)

[Haodong Li](https://graeme-lee.github.io/haodong/) , [Hengyong Yu](https://hengyongyu.wixsite.com/ctlab/untitled-cyjn)


![Model Architecture](figure_architecture.JPG)
![Figure1](attention.png)


This code is modified by [SiT(N Ma, S Xie.etl)](https://github.com/willisma/SiT). 

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
