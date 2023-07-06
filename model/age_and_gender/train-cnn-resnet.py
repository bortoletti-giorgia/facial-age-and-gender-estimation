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
from keras.models import Sequential
from keras import Model, Input
from keras import optimizers
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras import callbacks
from tensorflow import keras
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

import argparse
from age_groups import *

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--idprocess", help="id cluster DEI process", required=True)
parser.add_argument("--dsprefix", help="utface or utkface-wild", default='utkface')

args = parser.parse_args()
id_process = args.idprocess
ds_prefix = args.dsprefix

print("Id process: "+str(id_process))

# PATHS !!
ds_path = '/home/bortoletti/MODEL/'+ds_prefix+'/'
csv_path = '/home/bortoletti/MODEL/'+ds_prefix+'.csv'
results_folder = os.path.join(os.path.join(os.getcwd(), "results"), str(id_process))
if not os.path.exists(results_folder):
	os.mkdir(results_folder)

# TRAIN THE MODEL
df = pd.read_csv(csv_path)
n_tot_images = df.shape[0]
df.rename(columns = {'Unnamed: 0':'original-index'}, inplace = True)
# Change gender from string to integer
gender_mapper = {'male': 0, 'female': 1}
df = df.replace({"gender": gender_mapper})
df["age-group"]=df["age-group"].astype(str)

# Split in training and validation set
training_data, validation_data = train_test_split(df, test_size=0.3)

n_train = len(training_data)
n_val = len(validation_data)

print('No. of training image:', n_train)
print('No. of validation image:', n_val)

# set train and val data

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 64 # !!

img_size = 256
x_col = 'filename'
y_col = ['age', 'gender']

train_generator = train_datagen.flow_from_dataframe(training_data, 
                                                    directory = ds_path, 
                                                    x_col = x_col, 
                                                    y_col = y_col, 
                                                    target_size = (img_size, img_size), 
                                                    shuffle = False, class_mode="multi_output",
                                                    batch_size = batch_size)

val_generator = val_datagen.flow_from_dataframe(validation_data, 
                                                directory = ds_path, 
                                                x_col = x_col, 
                                                y_col = y_col, 
                                                target_size = (img_size, img_size),
						shuffle = False, class_mode="multi_output",
                                                batch_size = batch_size) # class_mode = 'multi_output',

# Create model
checkpoint = tf.keras.callbacks.ModelCheckpoint(results_folder+"/age_best_model.h5", 
                                                monitor='val_loss', 
                                                verbose=1, 
                                                save_best_only=True, 
                                                mode='min')
n_male, n_female = np.bincount(df['gender'])
initial_bias = np.log([n_female/n_male])
print("Init bias: ", initial_bias)
output_bias = tf.keras.initializers.Constant(initial_bias)

base_model = ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=None,
    pooling='avg'
)
for layer in base_model.layers:
    layer.trainable = False

age_model = Dense(12, activation='sigmoid', name='age_output')(base_model.output)
model = Model(inputs=base_model.input, outputs=age_model)

age_model = Dense(1, activation = 'linear', name='age_output', kernel_initializer='uniform')(base_model.output)
gender_model = Dense(1, activation = 'sigmoid', name='gender_output', bias_initializer=output_bias, kernel_initializer='uniform')(base_model.output)

model = Model(inputs=base_model.input, outputs=[age_model, gender_model])

model.summary()

#plot_model(model, to_file="model.jpg", show_shapes=True)

# TRAIN
epochs = 20 # !!
opt = keras.optimizers.Adam(learning_rate=0.001)
#opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
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
                    callbacks = [checkpoint], verbose=1)

print(history)
model.save(results_folder+"/model")

# PLOTTING
fig = plt.figure(figsize=(15,10))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=1)

fig.add_subplot(2,1,1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.legend()
plt.grid(True)
plt.xlabel('epoch')

fig.add_subplot(2,1,2)
plt.plot(history.history['sparse_categorical_accuracy'], label='train accuracy')
plt.plot(history.history['val_sparse_categorical_accuracy'], label='val accuracy')
plt.title('Age MAE')
plt.legend()
plt.grid(True)
plt.xlabel('epoch')

plt.savefig(results_folder+"/metrics.jpg")

#model = keras.models.load_model(results_folder+"/model")
model.evaluate(val_generator)
prediction = model.predict(val_generator)

print(prediction)

y_pred = prediction.argmax(axis=-1)

print("Predicted: ", y_pred)
y_pred = y_pred.astype('str')
#cm = confusion_matrix(validation_data['age-group'], y_pred)
#disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=AgeGroups.getLabels())
#disp.plot()

# Print some examples for PREDICTION
plt.figure(figsize=(10,10))
range_start = 0
range_end = range_start+9

for i in range(range_start, range_end):
    sample = validation_data.iloc[i]
    
    actual_age = sample['age-group']
    pred_age = y_pred[i]
    
    plt.subplot(3,3,i+1-range_start)
    plt.axis('off')
    plt.title('Actual: %s\nPred: %s' % (actual_age, pred_age))
    plt.imshow(Image.open(sample.filepath))

plt.savefig(results_folder+"/example.jpg")