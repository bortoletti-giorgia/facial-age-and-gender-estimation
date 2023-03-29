import argparse
import pandas as pd
import os
import cv2
from age_groups import *

# SET CSV folder, CSV filename and dataset name
parser = argparse.ArgumentParser()
parser.add_argument("--outdirgender", help="path output folders based on gender")
parser.add_argument("--outdirage", help="path output folders based on gender")
parser.add_argument("--filename", help="CSV dataset filename", required=True)
parser.add_argument("--source", help="path where CSV file is saved", required=True)
parser.add_argument("--nsamples", help="number of sample to keep for each label", type=int, required=False)

args = parser.parse_args()

csv_folder = args.source
csv_path = os.path.join(csv_folder, args.filename)
outdir_gender = args.outdirgender
outdir_age = args.outdirage
num_sample_range = args.nsamples

df = pd.read_csv(csv_path)
n_tot_images = df.shape[0]

take_all_samples = False
if num_sample_range is None:
    take_all_samples = True

# Create Gender folders
if outdir_gender is not None:
    if not os.path.exists(outdir_gender):
        os.mkdir(outdir_gender)

    # female
    dir_female_path = os.path.join(outdir_gender, "1")
    if not os.path.exists(dir_female_path):
        os.mkdir(dir_female_path)
    df_female = df[df['gender'] == 'female']

    if take_all_samples:
        num_sample_range = len(df_female)

    for i in range(num_sample_range):
        img = cv2.imread(df_female.iloc[i].filepath)
        name = df_female.iloc[i].filename
        cv2.imwrite(dir_female_path+"/"+name, img)

    # male
    dir_male_path = os.path.join(outdir_gender, "0")
    if not os.path.exists(dir_male_path):
        os.mkdir(dir_male_path)
    df_male = df[df['gender'] == 'male']

    if take_all_samples:
        num_sample_range = len(df_male)

    for i in range(num_sample_range):
        img = cv2.imread(df_male.iloc[i].filepath)
        name = df_male.iloc[i].filename
        cv2.imwrite(dir_male_path+"/"+name, img)

# Create Age folders
if outdir_age is not None:
    if not os.path.exists(outdir_age):
        os.mkdir(outdir_age)
    ranges = AgeGroups().getRanges()

    for i in range(len(ranges)):
        dir_range = os.path.join(outdir_age, str(i))
        if not os.path.exists(dir_range):
            os.mkdir(dir_range)
        df_i = df[df['age-group'] == i]

        if take_all_samples:
            num_sample_range = df_i.shape[0]

        for j in range(num_sample_range):
            img = cv2.imread(df_i.iloc[j].filepath)
            name = df_i.iloc[j].filename
            cv2.imwrite(dir_range+"/"+name, img)
