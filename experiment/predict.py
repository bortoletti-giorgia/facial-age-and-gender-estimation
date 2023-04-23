from tensorflow import keras
import tensorflow as tf
import argparse
import os
import numpy as np

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
sys.path.append('./anaconda3/envs/model/lib/python3.8/site-packages')

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--modelpath", help="Model path", type=str, required=True)
parser.add_argument("--resultfile", help="Txt file where to save the prediction", type=str, required=True)
parser.add_argument("--imagespath", help="Where the images on which to make the prediction are saved", type=str, required=True)

args = parser.parse_args()

model_path = args.modelpath
images_path = args.imagespath
result_file = args.resultfile

# Load model
model = keras.models.load_model(os.path.join(model_path+"/model_4"))

# Predict on each image
img_size = 256
filenames = os.listdir(images_path)

ages = []
genders = []

for filename in filenames:

	filepath = images_path+"/"+filename
	# https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
	gray = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2GRAY)
	fm = cv2.Laplacian(gray, cv2.CV_64F).var()
	# if the focus measure is less than the supplied threshold,
	# then the image should be considered "blurry"
	blurry = False
	if fm < 20:
		blurry = True
	
	if not blurry:
		img = tf.keras.utils.load_img(filepath, target_size = (img_size, img_size))
		img = tf.keras.utils.img_to_array(img)
		img = img * (1./255)
		img = tf.expand_dims(img, axis = 0)
		prediction = model.predict(img)
		prediction = np.round(prediction)
		age_pred = int(prediction[0])
		ages.append(age_pred)
		gender_pred = "male" if prediction[1] == 0 else "female"
		genders.append(gender_pred)
		'''
		age_real = int(filename.split("_")[0])
		gender_real = "male" if int(filename.split("_")[1]) == 0 else "female"
		print("AGE "+str(age_real)+" (real) "+str(age_pred)+" (predicted)")
		print("GENDER "+str(gender_real)+" (real) "+str(gender_pred)+" (predicted)")
		'''
		print(filename)
		print("AGE 24 (real) "+str(age_pred)+" (predicted)")
		print("GENDER female (real) "+str(gender_pred)+" (predicted)")
		# !!
		# show the image
		cv2.imshow(" ("+str(fm)+"), age "+str(age_pred)+", "+gender_pred, gray)
		cv2.waitKey(0)
cv2.destroyAllWindows() 
#cv2.waitKey(1)
# Analysis
print("Max age: ", max(ages))
print("Min age: ", min(ages))
print("Average: ", (sum(ages)/len(ages)))
print("Most gender: ", max(set(genders), key=genders.count))

final_age = sum(ages)/len(ages)
final_gender = max(set(genders), key=genders.count)

f = open(result_file, "w")
f.write(str(final_age)+","+str(final_gender))
f.close()
