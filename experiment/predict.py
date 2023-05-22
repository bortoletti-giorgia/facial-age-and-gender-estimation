'''
Make a prediction of age and gender given 'rgb' or 'grayscale' images of a person's face.
The model used, the image folder, the color mode and the TXT file where the prediction is to be saved must be passed as arguments. 
Arguments:
- modelpath: filepath where it is saved the "saved_model.pb" of a Keras model
- resulfile: filepath of the TXT file to save prediction in the format
	int(<age>),<gender> where gender is "female" or "male"
- imagespath: where the images on which to make the prediction are saved
- colormode: images color mode ("rgb" or "grayscale")
'''

from tensorflow import keras
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import argparse
import os
import numpy as np

import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--modelpath", help="Model path", type=str, required=True)
parser.add_argument("--resultfile", help="Txt file where to save the prediction", type=str, required=True)
parser.add_argument("--imagespath", help="Where the images on which to make the prediction are saved", type=str, required=True)
parser.add_argument("--colormode", help="rgb or grayscale", type=str, required=True)

args = parser.parse_args()

model_path = args.modelpath
images_path = args.imagespath
result_file = args.resultfile
colormode = args.colormode

# Load model
model = keras.models.load_model(model_path)

# Predict on each image
img_size = 256
filenames = os.listdir(images_path)

ages = []
genders = []

for filename in filenames:
	if colormode in filename:
		filepath = images_path+"/"+filename
		img = tf.keras.utils.load_img(filepath, target_size = (img_size, img_size), color_mode=colormode)
		img = tf.keras.utils.img_to_array(img)
		img = img * (1./255)
		img = tf.expand_dims(img, axis = 0)
		prediction = model.predict(img)
		prediction = np.round(prediction)
		age_pred = int(prediction[0])
		ages.append(age_pred)
		gender_pred = "male" if prediction[1] == 0 else "female"
		genders.append(gender_pred)

# Analysis
age_avg = sum(ages)/len(ages)
print("Max age: ", max(ages))
print("Min age: ", min(ages))
print("Average: ", age_avg)
print("Most gender: ", max(set(genders), key=genders.count))

final_age = np.round(age_avg)
final_gender = max(set(genders), key=genders.count)

f = open(result_file, "w")
f.write(str(final_age)+","+str(final_gender))
f.close()