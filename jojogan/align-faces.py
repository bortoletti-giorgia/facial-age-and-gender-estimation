from torchvision import transforms, utils
from PIL import Image
import math
import random
import os

import numpy as np
from torch import nn, autograd, optim
from torch.nn import functional as F
from tqdm import tqdm
import argparse

# JoJoGAN libraries
from model import *
from e4e_projection import projection as e4e_projection
from util import *

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--inputdir", help="directory where images are saved", required=True)
parser.add_argument("--outputdir", help="directory where aligned images will be saved", required=True)

args = parser.parse_args()

inputdir = args.inputdir
outputdir = args.outputdir

if not os.path.exists(outputdir):
	os.mkdir(outputdir)

device = 'cuda'

no_face_detected = 0

if "rosbag" in inputdir:
    subfolders = [f.path for f in os.scandir(inputdir) if f.is_dir() ]
    # read every rosbag folder
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
                        finaldir = os.path.join(outputdir, subfolder.split('/')[-1])
                        if not os.path.exists(finaldir):
                            os.mkdir(finaldir)
                        plt.imsave(os.path.join(finaldir, filename), get_image(aligned_face))
                    except AssertionError: # no face detected
                        pass
else: # folder of images
    files = os.listdir(inputdir)
    for filename in files:
        if filename[-4:] == '.jpg': # image
            filepath = os.path.join(inputdir, filename)
            if not os.path.exists(os.path.join(outputdir, filename)):
                name = strip_path_extension(filepath)+'.pt'
                # aligns and crops face from the rosbags image
                try:
                    aligned_face = align_face(filepath, output_size=256, transform_size=256)
                    my_w = e4e_projection(aligned_face, name, device).unsqueeze(0)
                    plt.imsave(os.path.join(outputdir, filename), get_image(aligned_face))
                except: # no face detected (AssertionError, RuntimeError, ValueError)
                    print("No face detected")
                    no_face_detected += 1
                    #pass

print("No face detected in "+str(no_face_detected)+" images.")
    