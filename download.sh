# FFHQ 256 DDPM
echo "Downloading FFHQ 256 DDPM model..."
gdown https://drive.google.com/uc?id=1BGwhRWUoguF-D8wlZ65tf227gp3cDUDh -O checkpoints/ffhq256.pt

# ImageNet 256 DDPM
echo "Downloading ImageNet 256 DDPM model..."
gdown https://drive.google.com/uc?id=1HAy7P19PckQLczVNXmVF-e_CRxq098uW -O checkpoints/imagenet256.pt

# FFHQ 256 LDM
echo "Downloading FFHQ 256 LDM model..."
wget https://ommer-lab.com/files/latent-diffusion/ffhq.zip -P ./checkpoints
unzip checkpoints/ffhq.zip -d ./checkpoints
mv checkpoints/model.ckpt checkpoints/ldm_ffhq256.pt
rm checkpoints/ffhq.zip

# ImageNet 256 LDM
echo "Downloading ImageNet 256 LDM model..."
wget https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt -P ./checkpoints/
mv checkpoints/model.ckpt checkpoints/ldm_imagenet256.pt

# Nonlinear blur model
echo "Downloading FFHQ nonlinear blur model..."
mkdir -p forward_operator/bkse/experiments/pretrained
gdown https://drive.google.com/uc?id=1vRoDpIsrTRYZKsOMPNbPcMtFDpCT6Foy -O forward_operator/bkse/experiments/pretrained/GOPRO_wVAE.pth

# Testing Dataset
echo "Downloading FFHQ test dataset..."
gdown https://drive.google.com/uc?id=1IzbnLWPpuIw6Z2E4IKrRByh6ihDE5QLO -O dataset/test-ffhq.zip
unzip dataset/test-ffhq.zip -d ./dataset
rm dataset/test-ffhq.zip

echo "Downloading ImageNet test dataset..."
gdown https://drive.google.com/uc?id=1pqVO-LYrRRL4bVxUidvy-Eb22edpuFCs -O dataset/test-imagenet.zip
unzip dataset/test-imagenet.zip -d ./dataset
rm dataset/test-imagenet.zip
