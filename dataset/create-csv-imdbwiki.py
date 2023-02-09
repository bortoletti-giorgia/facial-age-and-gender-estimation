# https://github.com/imdeepmind/processed-imdb-wiki-dataset/blob/master/mat.py

import argparse
import numpy as np
from scipy.io import loadmat
import pandas as pd
import datetime as date
from dateutil.relativedelta import relativedelta

# SET source path, output path, filename
parser = argparse.ArgumentParser()
parser.add_argument("--source", help="path where the dataset is")
parser.add_argument("--outdir", help="path where to save the CSV file")
parser.add_argument("--filename", help="CSV filename")

args = parser.parse_args()

ds_path = args.source
csv_path = args.outdir
filename_csv = args.filename

# ONLY ONCE: import data following the structure of given MAT files
# code adapted from: https://github.com/imdeepmind/processed-imdb-wiki-dataset/blob/master/mat.py

imdb_mat = ds_path + 'imdb_crop/imdb.mat'
wiki_mat =  ds_path + 'wiki_crop/wiki.mat'

imdb_data = loadmat(imdb_mat)
wiki_data = loadmat(wiki_mat)

del imdb_mat, wiki_mat

imdb = imdb_data['imdb']
wiki = wiki_data['wiki']

imdb_photo_taken = imdb[0][0][1][0]
imdb_full_path = imdb[0][0][2][0]
imdb_gender = imdb[0][0][3][0]
#imdb_name_celebrity = imdb[0][0][4][0]
imdb_face_location = imdb[0][0][5][0]
imdb_face_score1 = imdb[0][0][6][0]
imdb_face_score2 = imdb[0][0][7][0]

wiki_photo_taken = wiki[0][0][1][0]
wiki_full_path = wiki[0][0][2][0]
wiki_gender = wiki[0][0][3][0]
#wiki_name_celebrity = wiki[0][0][4][0]
wiki_face_location = wiki[0][0][5][0]
wiki_face_score1 = wiki[0][0][6][0]
wiki_face_score2 = wiki[0][0][7][0]

imdb_path = []
wiki_path = []

imdb_filename = []
wiki_filename = []

for path in imdb_full_path:
    imdb_path.append( ds_path + '/imdb_crop/' + path[0])
    imdb_filename.append(path[0].split('/')[1])

for path in wiki_full_path:
    wiki_path.append( ds_path + '/wiki_crop/' + path[0])
    wiki_filename.append(path[0].split('/')[1])

# gender
imdb_genders = []
wiki_genders = []

## imdb
for n in range(len(imdb_gender)):
    if imdb_gender[n] == 1:
        imdb_genders.append('male')
    else:
        imdb_genders.append('female')

## wiki
for n in range(len(wiki_gender)):
    if wiki_gender[n] == 1:
        wiki_genders.append('male')
    else:
        wiki_genders.append('female')
        
# when photo is taken
imdb_dob = []
wiki_dob = []

for file in imdb_filename:
    temp = file.split('_')[2]
    temp = temp.split('-')
    if len(temp[1]) == 1:
        temp[1] = '0' + temp[1]
    if len(temp[2]) == 1:
        temp[2] = '0' + temp[2]
    if temp[1] == '00':
        temp[1] = '01'
    if temp[2] == '00':
        temp[2] = '01'
    temp
    imdb_dob.append('-'.join(temp))

for file in wiki_filename:
    wiki_dob.append(file.split('_')[1])

# age
imdb_age = []
wiki_age = []

## imdb
for i in range(len(imdb_dob)):
    try:
        d1 = date.datetime.strptime(imdb_dob[i][0:10], '%Y-%m-%d')
        d2 = date.datetime.strptime(str(imdb_photo_taken[i]), '%Y')
        rdelta = relativedelta(d2, d1)
        diff = int(rdelta.years)
    except Exception as ex:
        print(ex)
        diff = -1
    imdb_age.append(diff)

## wiki
for i in range(len(wiki_dob)):
    try:
        d1 = date.datetime.strptime(wiki_dob[i][0:10], '%Y-%m-%d')
        d2 = date.datetime.strptime(str(wiki_photo_taken[i]), '%Y')
        rdelta = relativedelta(d2, d1)
        diff = int(rdelta.years)
    except Exception as ex:
        print(ex)
        diff = -1
    wiki_age.append(diff)

cols = ['age', 'gender', 'filepath', 'filename', 'face_score1', 'face_score2']

final_imdb = np.vstack((imdb_age, imdb_genders, imdb_path, imdb_filename, imdb_face_score1, imdb_face_score2)).T
final_wiki = np.vstack((wiki_age, wiki_genders, wiki_path, wiki_filename, wiki_face_score1, wiki_face_score2)).T

final_imdb_df = pd.DataFrame(final_imdb)
final_wiki_df = pd.DataFrame(final_wiki)

final_imdb_df.columns = cols
final_wiki_df.columns = cols

df = pd.concat((final_imdb_df, final_wiki_df))

df = df[df['face_score1'] != '-inf']
df = df[df['face_score2'] == 'nan']

df = df.drop(['face_score1', 'face_score2'], axis=1)

# Drop column with invalid ages
df['age'] = df['age'].astype('int')
df = df.drop(df[df.age < 0].index)

df = df.sample(frac=1)

# Create CSV file
df.to_csv(csv_path + filename_csv, index=False)
print("Total number of images: ", df.shape[0])
print("CSV file saved in:", csv_path + filename_csv)

