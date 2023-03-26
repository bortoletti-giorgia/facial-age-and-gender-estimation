import argparse
import pandas as pd 
from age_groups import *

# SET CSV folder and CSV filename
parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="path where CSV files are")
parser.add_argument("--filename", help="CSV filename")

args = parser.parse_args()

csv_folder = args.folder
csv_path = csv_folder + args.filename

# Age groups
ranges = AgeGroups().getRanges()

# Add column 'age-group' to CSV file based on 'age' column
df = pd.read_csv(csv_path)
df['label'] = ""

for i in range(len(ranges)):
    criteria = "age in @ranges["+str(i)+"]"
    df.loc[df.eval(criteria), 'age-group'] = i

# Save the updated CSV file
df.to_csv(csv_path, index=False)