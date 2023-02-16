import argparse
import pandas as pd
import os
import cv2

# SET CSV folder, CSV filename and dataset name
parser = argparse.ArgumentParser()
parser.add_argument("--outdirgender", help="path output folders based on gender")
parser.add_argument("--outdirage", help="path output folders based on gender")
parser.add_argument("--filename", help="CSV dataset filename")
parser.add_argument("--source", help="path where CSV file is saved")

args = parser.parse_args()

csv_folder = args.source
csv_path = csv_folder + args.filename
outdir_gender = args.outdirgender
dir_age_path = args.outdirage

df = pd.read_csv(csv_path)
n_tot_images = df.shape[0]

# Create Gender and Age folders
os.mkdir(outdir_gender)
dir_female_path = outdir_gender + "1"
dir_male_path = outdir_gender + "0"
os.mkdir(dir_female_path)
os.mkdir(dir_male_path)
os.mkdir(dir_age_path)

for i in range(n_tot_images):
    sample = df.loc[i]
    img = cv2.imread(sample.filepath)
    # gender
    if sample.gender == "female":
        cv2.imwrite(dir_female_path+"/"+str(i)+".jpg", img)
    elif sample.gender == "male":
        cv2.imwrite(dir_male_path+"/"+str(i)+".jpg", img)
    # age
    label = sample['label']
    dir_label_path = dir_age_path+str(label)
    if not os.path.isdir(dir_label_path):
        os.mkdir(dir_label_path)
    cv2.imwrite(dir_label_path+"/"+str(i)+".jpg", img)
