# https://www.kaggle.com/code/yflau17/age-gender-prediction-by-cnn

import os, shutil
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from PIL import Image

from keras.preprocessing.image import ImageDataGenerator
from keras import Model, Input
from keras import optimizers
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras import callbacks
from tensorflow import keras
import tensorflow as tf

import random
import argparse
from age_groups import *

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--idprocess", help="Id cluster DEI process", required=True)
parser.add_argument("--dsprefix", help="utkface or utkface-wild", default="utkface")
parser.add_argument("--inputdir", help="Path where dataset and CSV are", default="/home/bortoletti/MODEL/")
parser.add_argument("--outputdir", help="Results folder path", default="/home/bortoletti/MODEL/results/")

args = parser.parse_args()
id_process = args.idprocess
ds_prefix = args.dsprefix
root_path = args.inputdir
output_path = args.outputdir

print("Id process: "+str(id_process))

# PATHS !!
ds_path = os.path.join(root_path, ds_prefix+'-pepper/')
csv_path = os.path.join(root_path, ds_prefix+'-pepper.csv')
results_folder = os.path.join(output_path, str(id_process))
if not os.path.exists(results_folder):
	os.mkdir(results_folder)

# TRAIN THE MODEL
df = pd.read_csv(csv_path)
n_tot_images = df.shape[0]

gender_mapper = {'male': 0, 'female': 1}
df = df.replace({"gender": gender_mapper})
#df.info()

# Split in training and validation set
training_data, validation_data = train_test_split(df, test_size=0.3, shuffle=True)
validation_data.to_csv(results_folder+"/validation_data.csv") # save it to work locally

n_train = len(training_data)
n_val = len(validation_data)

print('No. of training image:', n_train)
print('No. of validation image:', n_val)

# Set training and validation data

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 64 # !!
y_cols = ['age', 'gender']
img_size = 256
x_col = 'filename'

train_generator = train_datagen.flow_from_dataframe(training_data, 
                                                    directory = ds_path, 
                                                    x_col = x_col, 
                                                    y_col = y_cols, 
                                                    target_size = (img_size, img_size),
                                                    class_mode="multi_output",
                                                    batch_size = batch_size)

val_generator = val_datagen.flow_from_dataframe(validation_data, 
                                                directory = ds_path, 
                                                x_col = x_col, 
                                                y_col = y_cols, 
                                                target_size = (img_size, img_size),
                                                class_mode="multi_output",
                                                shuffle=False,
                                                batch_size = batch_size)

# Create the model

# Set initial gender bias
# https://www.tensorflow.org/tutorials/structured_data/imbalanced_data?hl=it#clean_split_and_normalize_the_data
n_male, n_female = np.bincount(df['gender'])
initial_bias = np.log([n_female/n_male])
print("Init bias: ", initial_bias)
output_bias = tf.keras.initializers.Constant(initial_bias)

inputs = Input(shape=(256, 256, 3))

base_model = Conv2D(32, (3, 3), activation = 'relu')(inputs)
base_model = MaxPooling2D((2, 2))(base_model)
base_model = Conv2D(64, (3, 3), activation = 'relu')(base_model)
base_model = MaxPooling2D((2, 2))(base_model)
base_model = Conv2D(128, (3, 3), activation = 'relu')(base_model)
base_model = MaxPooling2D((2, 2))(base_model)
base_model = Dropout(0.5)(base_model)

age_model = base_model
age_model = Conv2D(256, (3, 3), activation = 'relu')(age_model)
age_model = MaxPooling2D((2, 2))(age_model)
age_model = Dropout(0.25)(age_model)
age_model = Conv2D(128, (3, 3), activation = 'relu')(age_model)
age_model = MaxPooling2D((2, 2))(age_model)
age_model = Dropout(0.25)(age_model)
age_model = Conv2D(64, (3, 3), activation = 'relu')(age_model)
age_model = MaxPooling2D((2, 2))(age_model)
age_model = Dropout(0.25)(age_model)
age_model = Flatten()(age_model)
age_model = Dense(256, activation = 'relu')(age_model)
age_model = Dense(128, activation = 'relu')(age_model)
age_model = Dense(64, activation = 'relu')(age_model)
age_model = Dense(32, activation = 'relu')(age_model)
age_model = Dense(1, activation = 'linear', name='age_output', kernel_initializer='uniform')(age_model)

gender_model = base_model
gender_model = Conv2D(256, (3, 3), activation = 'relu')(gender_model)
gender_model = MaxPooling2D((2, 2))(gender_model)
gender_model = Dropout(0.5)(gender_model)
gender_model = Flatten()(gender_model)
gender_model = Dense(128, activation = 'relu')(gender_model)
gender_model = Dense(64, activation = 'relu')(gender_model)
gender_model = Dense(32, activation = 'relu')(gender_model)
gender_model = Dense(1, activation = 'sigmoid', name='gender_output', bias_initializer=output_bias, kernel_initializer='uniform')(gender_model)

model = Model(inputs=inputs, outputs=[age_model, gender_model])

model.summary()
#plot_model(model, to_file="model.jpg", show_shapes=True)

# TRAIN
epochs = 20 # !!

earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 5, 
                                        restore_best_weights = True)

model.compile(loss={'age_output':'mse','gender_output':'binary_crossentropy'},
            optimizer='adam',
            metrics={'age_output':'mae','gender_output':'accuracy'})

history = model.fit(train_generator,
                    steps_per_epoch = n_train // batch_size, 
                    epochs = epochs,
                    batch_size = batch_size,
                    validation_data=val_generator,
                    validation_steps = n_val // batch_size,
                    callbacks = [earlystopping])
print(history)
model.save(results_folder+"/model")

# PLOTTING
fig = plt.figure(figsize=(15,10))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=1)

fig.add_subplot(3,2,1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.legend()
plt.grid(True)
plt.xlabel('epoch')

fig.add_subplot(3,2,3)
plt.plot(history.history['age_output_loss'], label='train loss')
plt.plot(history.history['val_age_output_loss'], label='val loss')
plt.title('Age Loss')
plt.legend()
plt.grid(True)
plt.xlabel('epoch')

fig.add_subplot(3,2,4)
plt.plot(history.history['age_output_mae'], label='train mae')
plt.plot(history.history['val_age_output_mae'], label='val mae')
plt.title('Age MAE')
plt.legend()
plt.grid(True)
plt.xlabel('epoch')

fig.add_subplot(3,2,5)
plt.plot(history.history['gender_output_loss'], label='train loss')
plt.plot(history.history['val_gender_output_loss'], label='val loss')
plt.title('Gender Loss')
plt.legend()
plt.grid(True)
plt.xlabel('epoch')

fig.add_subplot(3,2,6)
plt.plot(history.history['gender_output_accuracy'], label='train accuracy')
plt.plot(history.history['val_gender_output_accuracy'], label='val accuracy')
plt.title('Gender Accuracy')
plt.legend()
plt.grid(True)
plt.xlabel('epoch')
plt.savefig(results_folder+"/metrics.jpg")

# Evaluate the model
#model = keras.models.load_model(results_folder+'/model')
model.evaluate(val_generator)

# Make prediction
prediction = model.predict(val_generator)

print(prediction)

y_pred_age = np.round(prediction[0])
y_pred_gender = np.round(prediction[1])
y_pred_gender = y_pred_gender.astype('int')
y_pred_age = y_pred_age.astype('int')
validation_data["gender"]=validation_data["gender"].astype(int)
validation_data["age"]=validation_data["age"].astype(int)

# Precision
precision = tf.keras.metrics.Precision()
precision.update_state(validation_data["gender"], y_pred_gender)
print("Precision on gender: ", precision.result().numpy())
precision.update_state(validation_data["age"], y_pred_age)
print("Precision on age: ", precision.result().numpy())

# Recall
recall = tf.keras.metrics.Recall()
recall.update_state(validation_data["gender"], y_pred_gender)
print("Recall on gender: ", recall.result().numpy())
recall.update_state(validation_data["age"], y_pred_age)
print("Recall on age: ", recall.result().numpy())

# Confusion matrix GENDER
cm = confusion_matrix(validation_data['gender'], y_pred_gender, labels=np.unique(y_pred_gender))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_pred_gender))
disp.plot()
plt.savefig(results_folder+"/cm_gender.jpg")

# Confusion matrix AGE-GROUP
y_pred_groups = []

for pred in y_pred_age:
    y_pred_groups.append(AgeGroups().getGroupFromAge(pred))
    
cm = confusion_matrix(validation_data["age-group"], y_pred_groups)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_pred_groups))
disp.plot()
plt.savefig(results_folder+"/cm_age-groups.jpg")

# Print some example images with PREDICTION
plt.figure(figsize=(10,10))

indices = random.sample(np.arange(0,len(validation_data.index)).tolist(),9)

for j, i in enumerate(indices):
    sample = validation_data.iloc[i]
    
    actual_gender = "Female" if sample.gender==1 else "Male"
    pred_gender = "Female" if y_pred_gender[i]==1 else "Male"
    actual_age = sample['age']
    pred_age = y_pred_age[i]
    
    plt.subplot(3,3,j+1)
    plt.axis('off')
    plt.title('Actual: %s, %s\nPred: %s, %s' % (actual_gender, actual_age, pred_gender, pred_age))
    plt.imshow(Image.open(sample.filepath))

plt.savefig(results_folder+"/example.jpg")