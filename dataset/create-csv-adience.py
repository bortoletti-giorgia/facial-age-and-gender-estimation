# https://datasets.activeloop.ai/docs/ml/datasets/adience-dataset/
# https://talhassner.github.io/home/projects/Adience/Adience-data.html
# https://paperswithcode.com/dataset/adience

import argparse
import deeplake
import numpy as np
import pandas as pd

@deeplake.compute
def filter_ages(sample_in, ages_list):
    return sample_in.ages.data()['text'][0] in ages_list

@deeplake.compute
def filter_genders(sample_in, genders_list):
    return sample_in.genders.data()['text'][0] in genders_list 

# SET source path, output path, filename
parser = argparse.ArgumentParser()
parser.add_argument("--source", help="path where the dataset is")
parser.add_argument("--outdir", help="path where to save the CSV file")
parser.add_argument("--filename", help="CSV filename")

args = parser.parse_args()

ds_path = args.source
csv_path = args.outdir
filename_csv = args.filename

# ONLY ONCE: import data from deeplake

'''
ds = deeplake.load('hub://activeloop/mnist-train')
ds.tensors.keys()
ds_view.save_view(message = 'Samples with 0 and 8')
views = ds.get_views()
print(views)
print(len(ds_view))
'''

# Create a local copy of the dataset
#ds = deeplake.deepcopy('hub://activeloop/adience', './adience') 

# Load the local copy
#ds = deeplake.load('./adience')

# Load the remote copy
# https://datasets.activeloop.ai/docs/ml/datasets/adience-dataset/
ds = deeplake.load('hub://activeloop/adience')

# ds.visualize()
# ds.tensors.keys()

# Make a query to download all the images
gender_labels = ["m", "f", "u", "None"]
ds_all_data = ds.filter(filter_genders(gender_labels), scheduler = 'threaded', num_workers = 0)
n_tot_images = len(ds_all_data)

images = ds_all_data.images
ages = ds_all_data.ages.data()['text']
genders = ds_all_data.genders.data()['text'] 

# Make them arrays to save them in a DataFrame
ages_array = np.squeeze(ages)
genders_array = np.squeeze(genders)

# NOT DO THIS IF YOU HAVE ALREADY SAVED ALL THE IMAGES LOCALLY
# Save all images locally
# BE PATIENT: it takes some minutes (for me: 1 hour)
'''filepaths_array = []
for i, img in enumerate(images):
    img_to_save = Image.fromarray(img.numpy())
    temp = ds_path+str(i)+'.jpg'
    filepaths_array.append(temp)
    img_to_save = img_to_save.save(temp)
'''

# Define a dictionary containing images data
data = {'filename': [str(i)+'.jpg' for i in range(n_tot_images)],
        'gender': genders_array,
        'age': ages_array,
        'filepath': [ds_path+str(i)+'.jpg' for i in range(n_tot_images)]}

# Create a DataFrame
df = pd.DataFrame(data)
gender_mapper = {'m': 'male', 'f': 'female', 'u': 'neutral'}
df = df.replace({"gender": gender_mapper})

# Save CSV file
df.to_csv(csv_path + filename_csv, index=False)
print("Total number of images: ", df.shape[0])
print("CSV file saved in:", csv_path + filename_csv)