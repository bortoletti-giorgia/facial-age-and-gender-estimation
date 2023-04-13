from tensorflow import keras
import tensorflow as tf
import argparse
import os
import numpy as np

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--modelpath", help="Model path", default="C:/0_thesis/2_model/TESTING/both/19")
parser.add_argument("--inputdir", help="Where the images on which to make the prediction are saved", required=True)

args = parser.parse_args()

model_path = args.modelpath
images_path = args.inputdir

# Load model
model = keras.models.load_model(model_path+"/model")

# Predict on each image
img_size = 256
filenames = os.listdir(images_path)

for filename in filenames:
    img = tf.keras.utils.load_img(images_path+"/"+filename, target_size = (img_size, img_size))
    img = tf.keras.utils.img_to_array(img)
    img = img * (1./255)
    img = tf.expand_dims(img, axis = 0)
    prediction = model.predict(img)
    prediction = np.round(prediction)
    age_pred = int(prediction[0])
    gender_pred = "male" if prediction[1] == 0 else "female"

    age_real = int(filename.split("_")[0])
    gender_real = "male" if int(filename.split("_")[1]) == 0 else "female"
    print("AGE "+str(age_real)+" (real) "+str(age_pred)+" (predicted)")
    print("GENDER "+str(gender_real)+" (real) "+str(gender_pred)+" (predicted)")