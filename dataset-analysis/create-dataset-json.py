import argparse
import json
import os

# SET 
parser = argparse.ArgumentParser()
parser.add_argument("--source", help="path where dataset is saved")
parser.add_argument("--filename", help="JSON name")
args = parser.parse_args()

dataset_path = args.source
output_filename = args.filename


# Create dataset.json file with dataset labels
# This is useful to train the StyleGan2-ada-pytorch https://github.com/NVlabs/stylegan2-ada-pytorch with conditioning on labels.
# Follow the structure suggested in this issue: https://github.com/NVlabs/stylegan2-ada-pytorch/issues/111

'''{
    "labels":
        [
            ["folder1/1.jpg", 0], ["folder1/2.jpg", 0], ["folder1/3.jpg", 0], 
            ["folder2/4.jpg", 1], ["folder2/5.jpg", 1], ["folder2/6.jpg", 1], 
            ["folder3/7.jpg", 2], ["folder3/8.jpg", 2], ["folder3/9.jpg", 2], 
        ]
}
'''

# Create dictionary
data_dict = {}
data_dict['labels'] = []
label_counter = 0

for root, subdirs, files in os.walk(dataset_path):
    if len(subdirs) > 0:
        base_dir = root
        continue
    current_subdir = os.path.split(root)[1]
    for filename in files:
        file_path = current_subdir + "/" + filename
        #print('\t- file %s (full path: %s)' % (filename, file_path))
        data_dict['labels'].append([file_path, label_counter])

    label_counter += 1

# Save JSON file
with open(os.path.join(dataset_path, output_filename), 'w') as outfile:
    json.dump(data_dict, outfile)