'''
Create the (pretrained) model args.checkpointout+".pt"
from a set of style images contained in args.stylefolder or in "JoJoGAN/style_images" given their names in args.stylenames
and from StyleGAN trained using FFHQ dataset saved in "models/stylegan2-ffhq-config-f.pt".
Two parameters for intervening in the model configuration are:
- number of iteractions: args.iter
- alpha value: args.alpha
Alignment and cropping of faces are given by some utils file of JoJoGAN.
'''

# Initial setup
import torch
from torchvision import transforms, utils
from PIL import Image
import math
import random
import os
import gdown
import numpy as np
from torch import nn, autograd, optim
from torch.nn import functional as F
from tqdm import tqdm
from copy import deepcopy

import matplotlib.pyplot as plt
import argparse

# JoJoGAN Specific Import
from model import *
from e4e_projection import projection as e4e_projection
from util import *

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--checkpointin", help="pretrained stylegan2 with extension", default="stylegan2-ffhq-config-f.pt")
parser.add_argument("--ckptsize", help="image size of pretrained model", type=int, default=512)
parser.add_argument("--iter", help="number of iterations", type=int, default=500)
parser.add_argument("--alpha", help="alpha value", type=float, default=1)

parser.add_argument("--stylefolder", help="folder where images used for style are saved", type=str, default="style_images")
parser.add_argument("--testimagename", help="name for the test image with extension", required=True)
parser.add_argument("--stylenames", help="images used for style delimited with ,", type=str)
parser.add_argument("--checkpointout", help="name of your model to be created without extension", required=True)

args = parser.parse_args()

stylefolder = args.stylefolder
if args.stylenames is not None:
	stylenames = [str(item) for item in args.stylenames.split(',')]
else:
	stylenames = os.listdir(os.path.join(os.getcwd(), stylefolder))

checkpointin = args.checkpointin
source_filename = args.testimagename 
latent_dim = args.ckptsize
checkpointout = args.checkpointout
iter = args.iter
alpha = args.alpha

# Creating local folders for local content creation and management
os.makedirs('inversion_codes', exist_ok=True)
os.makedirs('style_images', exist_ok=True)
os.makedirs('style_images_aligned', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Download models
drive_ids = {
    "stylegan2-ffhq-config-f.pt": "1Yr7KuD959btpmcKGAUsbAk5rPjX2MytK",
    "e4e_ffhq_encode.pt": "1o6ijA3PkcewZvwJJ73dJ0fxhndn0nnh7",
    "restyle_psp_ffhq_encode.pt": "1nbxCIVw9H3YnQsoIPykNEFwWJnHVHlVd",
    "arcane_caitlyn.pt": "1gOsDTiTPcENiFOrhmkkxJcTURykW1dRc",
    "arcane_caitlyn_preserve_color.pt": "1cUTyjU-q98P75a8THCaO545RTwpVV-aH",
    "arcane_jinx_preserve_color.pt": "1jElwHxaYPod5Itdy18izJk49K1nl4ney",
    "arcane_jinx.pt": "1quQ8vPjYpUiXM4k1_KIwP4EccOefPpG_",
    "arcane_multi_preserve_color.pt": "1enJgrC08NpWpx2XGBmLt1laimjpGCyfl",
    "arcane_multi.pt": "15V9s09sgaw-zhKp116VHigf5FowAy43f",
    "sketch_multi.pt": "1GdaeHGBGjBAFsWipTL0y-ssUiAqk8AxD",
    "disney.pt": "1zbE2upakFUAx8ximYnLofFwfT8MilqJA",
    "disney_preserve_color.pt": "1Bnh02DjfvN_Wm8c4JdOiNV4q9J7Z_tsi",
    "jojo.pt": "13cR2xjIBj8Ga5jMO7gtxzIJj2PDsBYK4",
    "jojo_preserve_color.pt": "1ZRwYLRytCEKi__eT2Zxv1IlV6BGVQ_K2",
    "jojo_yasuho.pt": "1grZT3Gz1DLzFoJchAmoj3LoM9ew9ROX_",
    "jojo_yasuho_preserve_color.pt": "1SKBu1h0iRNyeKBnya_3BBmLr4pkPeg_L",
    "art.pt": "1a0QDEHwXQ6hE_FcYEyNMuv5r5UnRQLKT",
}

# from StyelGAN-NADA
class Downloader(object):
    def download_file(self, file_name):
        file_dst = os.path.join('models', file_name)
        file_id = drive_ids[file_name]
        if not os.path.exists(file_dst):
            print(f'Downloading {file_name}')
            gdown.download(id=file_id, output=file_dst, quiet=False)

downloader = Downloader()
#downloader.download_file('stylegan2-ffhq-config-f.pt')
#downloader.download_file('e4e_ffhq_encode.pt')

# CUDA
torch.backends.cudnn.benchmark = True
device = 'cuda' #@param ['cuda', 'cpu']

# Load StyleGAN model
# Already trained using FFHQ dataset - 70K faces
# latent_dim = 512
# Load original generator
original_generator = Generator(1024, latent_dim, 8, 2).to(device)
ckpt = torch.load('models/'+checkpointin, map_location=lambda storage, loc: storage)
original_generator.load_state_dict(ckpt["g_ema"], strict=False)
mean_latent = original_generator.mean_latent(10000)

# to be finetuned generator
generator = deepcopy(original_generator)
transform = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Source Image
filename = source_filename
filepath = f'test_input/{filename}'
name = strip_path_extension(filepath)+'.pt'
# aligns and crops face from the source image
aligned_face = align_face(filepath, output_size=256)
my_w = e4e_projection(aligned_face, name, device).unsqueeze(0)
plt.imsave("results/aligned_face.jpg", get_image(aligned_face))

# Create your own style
targets = []
latents = []
for i, name in enumerate(stylenames):
    style_path = os.path.join(stylefolder, name)
    assert os.path.exists(style_path), f"{style_path} does not exist!"

    name = strip_path_extension(name)
    if stylefolder.split("/")[-1] is not None:
        name += "_"+stylefolder.split("/")[-1]
        stylenames[i] = name

    # crop and align the face
    style_aligned_path = os.path.join('style_images_aligned', f'{name}.png')
    if not os.path.exists(style_aligned_path):
        style_aligned = align_face(style_path, output_size=256)
        style_aligned.save(style_aligned_path)
    else:
        style_aligned = Image.open(style_aligned_path).convert('RGB')

    # GAN invert
    style_code_path = os.path.join('inversion_codes', f'{name}.pt')
    if not os.path.exists(style_code_path):
        latent = e4e_projection(style_aligned, style_code_path, device)
    else:
        latent = torch.load(style_code_path)['latent']

    targets.append(transform(style_aligned).to(device))
    latents.append(latent.to(device))

targets = torch.stack(targets, 0)
latents = torch.stack(latents, 0)

target_im = utils.make_grid(targets, normalize=True, range=(-1, 1))
plt.imsave("results/style_reference.jpg", get_image(target_im))

# Finetune StyleGAN
#alpha =  1 #@param {type:"slider", min:0, max:1, step:0.1}
alpha = 1-alpha

# Tries to preserve color of original image by limiting family of allowable transformations. Set to false if you want to transfer color from reference image. This also leads to heavier stylization
preserve_color = False 
# Number of finetuning steps. Different style reference may require different iterations. Try 200~500 iterations.
num_iter = iter

# load discriminator for perceptual loss
discriminator = Discriminator(1024, 2).eval().to(device)
ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
discriminator.load_state_dict(ckpt["d"], strict=False)

# reset generator
del generator
generator = deepcopy(original_generator)
g_optim = optim.Adam(generator.parameters(), lr=2e-3, betas=(0, 0.99))

# Which layers to swap for generating a family of plausible real images -> fake image
if preserve_color:
    id_swap = [9,11,15,16,17]
else:
    id_swap = list(range(7, generator.n_latent))

for idx in tqdm(range(num_iter)):
    mean_w = generator.get_latent(torch.randn([latents.size(0), latent_dim]).to(device)).unsqueeze(1).repeat(1, generator.n_latent, 1)
    in_latent = latents.clone()
    in_latent[:, id_swap] = alpha*latents[:, id_swap] + (1-alpha)*mean_w[:, id_swap]

    img = generator(in_latent, input_is_latent=True)

    with torch.no_grad():
        real_feat = discriminator(targets)
    fake_feat = discriminator(img)

    loss = sum([F.l1_loss(a, b) for a, b in zip(fake_feat, real_feat)])/len(fake_feat)
    
    g_optim.zero_grad()
    loss.backward()
    g_optim.step()

# Generate results
n_sample =  5
seed = 3000
torch.manual_seed(seed)
with torch.no_grad():
    generator.eval()
    z = torch.randn(n_sample, latent_dim, device=device)
    my_sample = generator(my_w, input_is_latent=True)

style_images = []
for name in stylenames:
    style_path = f'style_images_aligned/{strip_path_extension(name)}.png'
    style_image = transform(Image.open(style_path))
    style_images.append(style_image)

# Save the model checkpoint to the Local Disk
torch.save(generator.state_dict(), '/home/bortoletti/JoJoGAN/models/'+checkpointout+'.pt')

# Save final results
face = transform(aligned_face).to(device).unsqueeze(0)
style_images = torch.stack(style_images, 0).to(device)
plt.imsave("style_images_aligned/"+checkpointout+'.jpg', get_image(utils.make_grid(style_images, normalize=True, range=(-1, 1))))

my_output = torch.cat([style_images, face, my_sample], 0)
plt.imsave("results/final_sample.jpg", get_image(utils.make_grid(my_output, normalize=True, range=(-1, 1))))
