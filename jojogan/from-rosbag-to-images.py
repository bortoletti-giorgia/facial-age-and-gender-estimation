import bagpy
from bagpy import bagreader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="directory where rosbags are and where to save the images", required=True)

args = parser.parse_args()

folder = args.dir 

#folder = 'C:/SharedFolder/ROSBAG-style-reference/'
entries = os.listdir(folder)
for bagfile in entries:
    b = bagreader(folder+bagfile)
    df = pd.read_csv(b.message_by_topic('/pepper_robot/camera/front/camera/image_raw'))
    bagfolder = folder+bagfile[:-4]
    for i in range(0, len(df)):
        # raw string appears as: 'b\' \\\'\\n"*\\x0c$\\\'\\x14\\x1f#...'
        raw_string = df['data'][i]

        # convert to byte string with escape characters included
        byte_string = raw_string[2:-1].encode('latin1')

        # remove escaped characters
        escaped_string = byte_string.decode('unicode_escape')

        # convert back to byte string without escaped characters
        byte_string = escaped_string.encode('latin1')

        # convert string to numpy array
        # this will throw a warning to use np.frombuffer
        nparr = np.frombuffer(byte_string, np.uint8)

        # convert to 3 channel rgb image array of (H x W x 3)
        image = nparr.reshape((240, 320, -1))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # show image in matplotlib
        plt.imsave(bagfolder+'/'+str(i)+'.jpg', rgb)
        
print("End.")