import argparse
import cv2
import os

# SET 
parser = argparse.ArgumentParser()
parser.add_argument("--source", help="path where dataset is saved", required=True)
parser.add_argument("--outdir", help="path where to save resized images", required=True)
parser.add_argument("--imagesize", help="image size should be power-of-two", type=int, required=True)
args = parser.parse_args()

dataset_path = args.source
output_folder = args.outdir
im_size = args.imagesize   # Input image resolution must be a power-of-two otherwise you will get error. 

images = []

for root, subdirs, files in os.walk(dataset_path):
    if len(subdirs) > 0:
        base_dir = root
        continue
    current_subdir = os.path.split(root)[1]
    for filename in files:
        in_filepath = os.path.join(dataset_path + current_subdir, filename)
        img = cv2.imread(in_filepath)  # reading that image as array
        img = cv2.resize(img, (im_size, im_size))
        #print(img.shape)
        images.append(img)
        # Save the image in Output Folder
        outdir = output_folder + current_subdir
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        cv2.imwrite(outdir + '/' + str(filename), img)
