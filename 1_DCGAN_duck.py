"""
DCGAN training script (PyTorch).

@Author : Cédric Ludwigs
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

# -----------------
# Config
# -----------------
data_root = 'data/duck_resized'
models_dir = 'models/duck_dcgan_v4'
os.makedirs(models_dir, exist_ok=True)
os.makedirs(os.path.join(models_dir, 'inference'), exist_ok=True)

image_size = 128
channels = 3  # RGB
latent_dim = 100
batch_size = 32
epochs = 300
lr = 0.0002
beta1 = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------
# Dataset + DataLoader
# -----------------
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * channels, [0.5] * channels),  # map to [-1,1]
])

if not os.path.exists(data_root):
    print(f"Dossier dataset '{data_root}' introuvable. Créez-le et ajoutez des images.")
    sys.exit(1)

dataset = datasets.ImageFolder(root=data_root, transform=transform)
if len(dataset) == 0:
    print(f"Aucune image trouvée dans '{data_root}'. Vérifiez que les fichiers sont présents dans un sous-dossier.")
    sys.exit(1)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == 'cuda'))

# -----------------
# Helper: weights init
# -----------------
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# -----------------
# Generator
# -----------------
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        # Start from (nz x 1 x 1) and upscale to (nc x 128 x 128)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),  # 4x4
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),  # 32x32
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),  # 64x64
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf // 2, nc, 4, 2, 1, bias=False),  # 128x128
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)

# -----------------
# Discriminator
# -----------------
class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf // 2, 4, 2, 1, bias=False),  # 64
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf // 2, ndf, 4, 2, 1, bias=False),  # 32
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # 16
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),  # 8
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),  # 4
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),  # output 1x1
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

# -----------------
# Label Helpers
# -----------------
""" def real_label(size, device):
    return torch.full((size,), 1., dtype=torch.float, device=device) """

def real_label(size, device):
    return torch.empty(size, device=device).uniform_(0.8, 1.0)

def fake_label(size, device):
    return torch.full((size,), 0., dtype=torch.float, device=device)

# -----------------
# Main Training Loop
# -----------------
if __name__ == '__main__':
    ngf = 64
    ndf = 64
    
    # Build models
    netG = Generator(nz=latent_dim, ngf=ngf, nc=channels).to(device)
    netD = Discriminator(nc=channels, ndf=ndf).to(device)

    netG.apply(weights_init)
    netD.apply(weights_init)

    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Fixed noise for monitoring
    fixed_noise = torch.randn(16, latent_dim, 1, 1, device=device)

    print(f"Démarrage de l'entraînement sur : {device}, taille du dataset : {len(dataset)}")

    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device)
            b_size = imgs.size(0)

            # ---------------------------
            # 1. Train Discriminator
            # ---------------------------
            netD.zero_grad()
            
            # Train with real images
            labels = real_label(b_size, device)
            output = netD(imgs)
            errD_real = criterion(output, labels)
            errD_real.backward()

            # Train with fake images
            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake = netG(noise)
            labels = fake_label(b_size, device)
            output = netD(fake.detach())
            errD_fake = criterion(output, labels)
            errD_fake.backward()
            
            errD = errD_real + errD_fake
            optimizerD.step()

            # ---------------------------
            # 2. Train Generator
            # ---------------------------
            netG.zero_grad()
            # Le générateur veut que le discriminateur classe ses fakes comme "vrais" (1)
            labels = real_label(b_size, device)  
            output = netD(fake)
            errG = criterion(output, labels)
            errG.backward()
            optimizerG.step()

            # Affichage de la progression
            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{i}/{len(loader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}")

        # Sauvegarde d'échantillons d'images générées
        with torch.no_grad():
            sample = netG(fixed_noise).detach().cpu()
            save_image((sample + 1) / 2.0, os.path.join(models_dir, f'sample_epoch_{epoch+1:03d}.png'), nrow=4)

        # Sauvegarde du modèle générateur à la fin de chaque epoch
        torch.save(netG.state_dict(), os.path.join(models_dir, 'dcgan_generator.pth'))

    print(f"Entraînement terminé. Générateur sauvegardé dans : {os.path.join(models_dir, 'dcgan_generator.pth')}")

