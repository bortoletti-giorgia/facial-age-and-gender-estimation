import torch
from torchvision import transforms, utils
from PIL import Image
import math
import random
import os

import numpy as np
from torch import nn, autograd, optim
from torch.nn import functional as F
from tqdm import tqdm
import wandb

from model import *
from e4e_projection import projection as e4e_projection
from util import *

folder = '/home/bortoletti/rosbag/'
subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]

outdir = '/home/bortoletti/rosbag/aligned-faces/'

if not os.path.exists(outdir):
	os.mkdir(outdir)

device = 'cuda'

for i, subfolder in enumerate(subfolders):
    if subfolder.split('/')[-1][0:4] == '2023':
        files = os.listdir(subfolder)
        for filename in files:
            if filename[-4:] == '.jpg': # image
                filepath = os.path.join(subfolder, filename)
                name = strip_path_extension(filepath)+'.pt'
                # aligns and crops face from the rosbags image
                try:
                    aligned_face = align_face(filepath, output_size=256, transform_size=256)
                    my_w = e4e_projection(aligned_face, name, device).unsqueeze(0)
                    plt.imsave(os.path.join(outdir, str(i)+'_'+filename), get_image(aligned_face))
                except AssertionError: # no face detected
                    pass