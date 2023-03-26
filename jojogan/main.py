'''
Transfer the style from the pretrained model args.modelname+".pt"
to a set of images contained in args.testfolder or in "JoJoGAN/test_input" given their names in args.testimagenames.
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
parser.add_argument("--modelname", help="where the pretrained model is saved without extension", required=True)
parser.add_argument("--testfolder", help="folder where images used for style are saved", type=str, default="test_input")
parser.add_argument("--testimagenames", help="name for the test image with extension delimited with ,", type=str)

args = parser.parse_args()
testfolder = args.testfolder
if args.testimagenames is not None:
	testimagenames = [str(item) for item in args.testimagenames.split(',')]
else:
	files = os.listdir(os.path.join(os.getcwd(), testfolder))
	testimagenames = []
	for f in files:
		if f[-4:] == ".jpg" or f[-5:] == ".jpeg" or f[-4:] == ".png": 
			testimagenames.append(f)
#print(testimagenames)
pretrained = args.modelname

# Creating local folders for local content creation and management
os.makedirs('inversion_codes', exist_ok=True)
os.makedirs('style_images', exist_ok=True)
os.makedirs('style_images_aligned', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# CUDA
torch.backends.cudnn.benchmark = True
device = 'cuda' #@param ['cuda', 'cpu']

# Load original generator from StyleGAN trained using FFHQ dataset
latent_dim = 512
original_generator = Generator(1024, latent_dim, 8, 2).to(device)
ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
original_generator.load_state_dict(ckpt["g_ema"], strict=False)
mean_latent = original_generator.mean_latent(10000)

# to be finetuned generator
generator = deepcopy(original_generator)
'''
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
'''

# Source Image
for filename in testimagenames:
    filepath = f'{testfolder}/{filename}'
    name = strip_path_extension(filepath)+'.pt'

    # Align and crop face from the source image
    try:
        aligned_face = align_face(filepath, output_size=256)
    except AssertionError as err:
        print(err)
    else:
        my_w = e4e_projection(aligned_face, name, device).unsqueeze(0)

        plt.imsave("results/aligned_face.jpg", get_image(aligned_face))

        # Test with pretrained style
        preserve_color = False
        ckpt = f'{pretrained}.pt'
        ckpt = torch.load(os.path.join('models', ckpt), map_location=lambda storage, loc: storage)
        generator.load_state_dict(ckpt, strict=False)

        # Generate results
        n_sample = 5
        seed = 3000
        torch.manual_seed(seed)
        with torch.no_grad():
            generator.eval()
            z = torch.randn(n_sample, latent_dim, device=device)
            sample_output = generator(my_w, input_is_latent=True)

        # Save output image
        '''# Save output: [style_image, test_input, final_sample]
        style_path = f'style_images_aligned/{pretrained}.jpg'
        style_image = transform(Image.open(style_path)).unsqueeze(0).to(device)
        my_sample = transform(aligned_face).unsqueeze(0).to(device)
        final_output = torch.cat([style_image, my_sample, sample_output], 0)
        plt.imsave("results/final_sample.jpg", get_image(utils.make_grid(final_output, normalize=True, range=(-1, 1))))
        '''

        root_result_folder = os.path.join(os.path.join(os.getcwd(), "results"), testfolder.split("/")[1]) # /home/bortoletti/JoJoGAN/results/test_input/utk-test/
        if not os.path.exists(os.path.join(root_result_folder)):
            os.mkdir(root_result_folder)
        result_subfolder = os.path.join(root_result_folder, testfolder.split("/")[2]) # /home/bortoletti/JoJoGAN/results/test_input/utk-test/<age-group>
        #result_subfolder = os.path.join(root_result_folder, pretrained.split("_")[1]) # /home/bortoletti/JoJoGAN/results/test_input/utk-test/<middle-age>-<config>
        if not os.path.exists(os.path.join(result_subfolder)):
            os.mkdir(result_subfolder)
            print("Creating images in "+result_subfolder)

        final_output = torch.cat([sample_output], 0)
        final_output_path = result_subfolder+"/"+filename
        plt.imsave(final_output_path, get_image(utils.make_grid(sample_output, normalize=True, range=(-1, 1))))
	
        # resize to 256x256
        im = Image.open(final_output_path)
        im = im.resize((256, 256))
        im.save(final_output_path)