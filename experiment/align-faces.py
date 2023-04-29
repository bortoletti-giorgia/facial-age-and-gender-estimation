from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
if './anaconda3/envs/jojo/lib/python3.7/site-packages' in sys.path:
    sys.path.append('./anaconda3/envs/jojo/lib/python3.7/site-packages')

# JoJoGAN libraries
from util import *

parser = argparse.ArgumentParser()
parser.add_argument("--inputdir", help="directory where images are saved", required=True)
parser.add_argument("--outputdir", help="directory where aligned images will be saved", required=True)

args = parser.parse_args()

inputdir = args.inputdir
outputdir = args.outputdir

if not os.path.exists(outputdir):
	os.mkdir(outputdir)

no_face_detected = 0

files = os.listdir(inputdir)
for filename in files:
    if filename[-4:] == '.jpg': # image
        filepath = os.path.join(inputdir, filename)
        if not os.path.exists(os.path.join(outputdir, filename)):
            # aligns and crops face from the rosbags image
            try:
                aligned_face = align_face(filepath, output_size=256, transform_size=256)
                plt.imsave(os.path.join(outputdir, "rgb_"+filename), get_image(aligned_face))
                gray = cv2.cvtColor(cv2.imread(os.path.join(outputdir, "rgb_"+filename)), cv2.COLOR_BGR2GRAY)
                cv2.imwrite(os.path.join(outputdir, "grayscale_"+filename), gray)
            except: # no face detected (AssertionError, RuntimeError, ValueError)
                print("No face detected")
                no_face_detected += 1
                #pass

print("No face detected in "+str(no_face_detected)+" images.")
    