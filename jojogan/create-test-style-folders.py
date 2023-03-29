import argparse
import os
import cv2

'''
parser = argparse.ArgumentParser()
parser.add_argument("--samdir", help="path where output images of SAM are saved")
parser.add_argument("--originaldir", help="path where original PEPPER images are saved")
parser.add_argument("--outdir", help="path where to save output")

args = parser.parse_args()

samdir = args.samdir
originaldir = args.originaldir
outdir = args.outdir
'''
N_SHOTS = 6
N_CONFIG = N_SHOTS-1

samdir="C:/0_thesis/1_stylegan/sam-model/results-"+str(N_SHOTS)+"shots/inference_results"
originaldir="C:/0_thesis/1_stylegan/sam-model/results-"+str(N_SHOTS)+"shots/ori"
outdir="C:/0_thesis/1_stylegan/jojogan-model/test-style-sam/style_images-"+str(N_SHOTS)+"shots"

sam_dirs = [d.path for d in os.scandir(samdir) if d.is_dir()]

original_images = [d.path for d in os.scandir(originaldir)]
n_images = len(original_images)

# 4/5 configurations:
# original-sam-sam-sam-sam(-sam) images
# ori-ori-sam-sam-sam(-sam)
# ori-ori-ori-sam-sam(-sam)
# ori-ori-ori-ori-sam(-sam)
# (ori-ori-ori-ori-ori-sam)
end_ori = 1 # how many original images this configuration considers
for config in range(N_CONFIG):
    # for each age-group
    for dir in sam_dirs:
        middle = dir.split("\\")[-1]
        outdir_middle = outdir + "/" + str(middle)+"-"+str(config)
        if not os.path.exists(outdir_middle):
            os.mkdir(outdir_middle)

        sam_images = [d.path for d in os.scandir(dir)]

        for i in range(end_ori):
            img = cv2.imread(original_images[i])
            name = original_images[i].split("\\")[-1]
            cv2.imwrite(os.path.join(outdir_middle, name), img)
        for j in range(end_ori, n_images):
            img = cv2.imread(sam_images[j])
            name = sam_images[j].split("\\")[-1]
            cv2.imwrite(os.path.join(outdir_middle, name), img)
    end_ori += 1




