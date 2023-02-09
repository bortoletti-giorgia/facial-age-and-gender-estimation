import argparse
import pandas as pd 

# SET CSV folder and CSV filename
parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="path where CSV files are")
parser.add_argument("--filename", help="CSV filename")

args = parser.parse_args()

csv_folder = args.folder
csv_path = csv_folder + args.filename

# Age groups
ranges = [range(0,3), range(3,13), range(13, 20), range(20, 30),
         range(30, 40), range(40, 50), range(50, 60), range(60, 70),
         range(70, 80), range(80, 90), range(90, 100), range(100,117)]


# Add column 'age-group' to CSV file based on 'age' column
df = pd.read_csv(csv_path)
df['label'] = ""

for i in range(len(ranges)):
    criteria = "age in @ranges["+str(i)+"]"
    df.loc[df.eval(criteria), 'label'] = i

# Save the updated CSV file
df.to_csv(csv_path, index=False)