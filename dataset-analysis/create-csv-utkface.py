

import argparse
import pandas as pd
import os


# SET source path, output path, filename
parser = argparse.ArgumentParser()
parser.add_argument("--source", help="path where the dataset is")
parser.add_argument("--outdir", help="path where to save the CSV file")
parser.add_argument("--filename", help="CSV filename")

args = parser.parse_args()

ds_path = args.source
csv_path = args.outdir
filename_csv = args.filename

# ONLY ONCE: create CSV file from local dataset
# Each image is saved as: [age] _ [gender] _ [race] _ [date&time].jpg

ds_tot_images = 0
for dirname, _, filenames in os.walk(ds_path):
    for filename in filenames:
        ds_tot_images += 1
        splitted = filename.split('_')
        age = splitted[0]
        gender = splitted[1]
        if gender != str(1) and gender != str(0):
            print("Image with wrong gender: "+filename)
        race = splitted[2]
        #print(os.path.join(dirname, filename))
        #break

# Save DataFrame structure with all the images
df = pd.DataFrame(filenames, columns = ['filename'] )
df['filepath'] = df.filename.apply(lambda x: ds_path + x )
df['age'] = df.filename.apply(lambda x: int(x.split('_')[0]))
df['gender'] = df.filename.apply(lambda x: int(x.split('_')[1]))
df['ethnicity'] = df.filename.apply(lambda x: int(x.split('_')[-2]))

gender_mapper = {0: 'male', 1: 'female'}
ethnicity_mapper = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others'}
df = df.replace({"gender": gender_mapper})
df = df.replace({"ethnicity": ethnicity_mapper})

# Save DataFrame to CSV file
df.to_csv(csv_path + filename_csv, index=False)
print("Total number of images: ", df.shape[0])
print("CSV file saved in:", csv_path + filename_csv)