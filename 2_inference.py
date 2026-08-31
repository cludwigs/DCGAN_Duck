

import os
import math
import argparse
import time
import torch
import torch.nn as nn
from torchvision.utils import save_image

# Default configuration must match training script
latent_dim = 100
image_size = 128
channels = 3
model_path = "duck_dcgan_v4"
os.makedirs(os.path.join("models", model_path, 'inference'), exist_ok=True)


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf // 2, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)


def load_generator(model_path, device):
    ngf = 64
    netG = Generator(nz=latent_dim, ngf=ngf, nc=channels).to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    state = torch.load(model_path, map_location=device)
    # state may be a state_dict or a dict containing more keys
    if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
        state_dict = state['state_dict']
    else:
        state_dict = state

    # Handle DataParallel prefixes
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state[k[len('module.'):]] = v
        else:
            new_state[k] = v

    netG.load_state_dict(new_state)
    netG.eval()
    return netG


def generate_images(netG, device, num_images=1, out_path=None, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    n = num_images
    noise = torch.randn(n, latent_dim, 1, 1, device=device)
    with torch.no_grad():
        fake = netG(noise).cpu()

    # fake is in range [-1,1], map to [0,1]
    imgs = (fake + 1.0) / 2.0

    if out_path is None:
        timestamp = int(time.time())
        out_path = f"models/{model_path}/inference/inference_{timestamp}.png"

    # If multiple images, save as a grid
    save_image(imgs, out_path, nrow=int(math.sqrt(max(1, n))))
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=f'models/{model_path}/dcgan_generator.pth', help='path to generator checkpoint')
    parser.add_argument('--num', type=int, default=1, help='number of images to generate')
    parser.add_argument('--out', type=str, default=None, help='output image path (png)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        netG = load_generator(args.model, device)
    except Exception as e:
        print('Error loading generator:', e)
        raise

    out_file = args.out or f"models/{model_path}/inference/inference_{int(time.time())}.png"
    out_path = generate_images(netG, device, num_images=args.num, out_path=out_file, seed=args.seed)
    print('Saved generated image to', out_path)


if __name__ == '__main__':
    main()
