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

import random
import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--idprocess", help="id cluster DEI process", required=True)
parser.add_argument("--dsprefix", help="utface or utkface-wild", default='utkface')

args = parser.parse_args()
id_process = args.idprocess
ds_prefix = args.dsprefix

print("Id process: "+str(id_process))

# PATHS !!
ds_path = '/home/bortoletti/MODEL/'+ds_prefix+'-pepper/'
csv_path = '/home/bortoletti/MODEL/'+ds_prefix+'-pepper.csv'
results_folder = os.path.join(os.path.join(os.getcwd(), "results"), str(id_process))
if not os.path.exists(results_folder):
	os.mkdir(results_folder)

# TRAIN THE MODEL
df = pd.read_csv(csv_path)
n_tot_images = df.shape[0]

gender_mapper = {'male': 0, 'female': 1}
df = df.replace({"gender": gender_mapper})
df['gender']=df['gender'].astype(str)

# Split in training and validation set
training_data, validation_data = train_test_split(df, test_size=0.3)
validation_data.to_csv(results_folder+"/validation_data.csv") # save it to work locally

n_train = len(training_data)
n_val = len(validation_data)

print('No. of training image:', n_train)
print('No. of validation image:', n_val)

# Weights for imbalanced training set
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(training_data['gender']), y=training_data['gender'])
class_weights = dict(enumerate(class_weights))

# Set train and val data generator

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 64 # !!
img_size = 256
x_col = 'filename'
y_col = 'gender' 

train_generator = train_datagen.flow_from_dataframe(training_data, 
                                                    directory = ds_path, 
                                                    x_col = x_col, 
                                                    y_col = y_col, 
                                                    target_size = (img_size, img_size), 
                                                    class_mode="binary",
                                                    batch_size = batch_size)

val_generator = val_datagen.flow_from_dataframe(validation_data, 
                                                directory = ds_path, 
                                                x_col = x_col, 
                                                y_col = y_col, 
                                                target_size = (img_size, img_size),
                                                class_mode="binary",
                                                shuffle = False,
                                                batch_size = batch_size)
# Create model

# https://www.tensorflow.org/tutorials/structured_data/imbalanced_data?hl=it#clean_split_and_normalize_the_data
n_male, n_female = np.bincount(df['gender'])
initial_bias = np.log([n_female/n_male])
print("Init bias: ", initial_bias)
output_bias = tf.keras.initializers.Constant(initial_bias)

inputs = Input(shape=(256, 256, 3))

gender_model = Conv2D(256, (3, 3), activation = 'relu')(inputs)
gender_model = MaxPooling2D((2, 2))(gender_model)
gender_model = Dropout(0.5)(gender_model)
gender_model = Flatten()(gender_model)
gender_model = Dense(128, activation = 'relu')(gender_model)
gender_model = Dense(64, activation = 'relu')(gender_model)
gender_model = Dense(32, activation = 'relu')(gender_model)
gender_model = Dense(1, activation = 'sigmoid', name='gender_output', bias_initializer=output_bias)(gender_model)

model = Model(inputs=inputs, outputs=gender_model)

model.summary()
#plot_model(model, to_file=results_folder+"/model.jpg", show_shapes=True)

# TRAIN
epochs = 20 # !!

opt = keras.optimizers.Adam(learning_rate=0.001)
#opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
#opt = keras.optimizers.SGD(lr=1e-4, momentum=0.9)

earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 5, 
                                        restore_best_weights = True)

model.compile(loss={'gender_output':'binary_crossentropy'}, 
            optimizer=opt,
            metrics={'gender_output':'accuracy'}) # !! optimizer="adam", categorical_crossentropy

history = model.fit(train_generator,
                    steps_per_epoch = n_train // batch_size, 
                    epochs = epochs,
                    batch_size = batch_size,
                    validation_data=val_generator,
                    validation_steps = n_val // batch_size,
                    callbacks = [earlystopping], 
                    verbose=1,
                    class_weight=class_weights)

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
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Gender accuracy')
plt.legend()
plt.grid(True)
plt.xlabel('epoch')

plt.savefig(results_folder+"/"+"metrics.jpg")

#''' !!
#model = keras.models.load_model(results_folder+'/model')
model.evaluate(val_generator)
prediction = model.predict(val_generator)

print(prediction)

y_pred = np.round(prediction)

print("Predicted: ", y_pred)

y_pred = y_pred.astype(int)
validation_data[y_col]=validation_data[y_col].astype(int)

# Precision
precision = tf.keras.metrics.Precision()
precision.update_state(validation_data[y_col], y_pred)
print("Precision: ", precision.result().numpy())

# Recall
recall = tf.keras.metrics.Recall()
recall.update_state(validation_data[y_col], y_pred)
print("Recall: ", recall.result().numpy())

# Confusion matrix
cm = confusion_matrix(validation_data[y_col], y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(validation_data[y_col]))
disp.plot()
plt.savefig(results_folder+"/cm.jpg")

# Print some examples for PREDICTION
plt.figure(figsize=(10,10))

indices = random.sample(np.arange(0,len(validation_data.index)).tolist(),9)

for j, i in enumerate(indices):
    sample = validation_data.iloc[i]
    
    actual_gender = "Female" if sample.gender==1 else "Male"
    pred_gender = "Female" if y_pred[i]==1 else "Male"
    
    plt.subplot(3,3,j+1)
    plt.axis('off')
    plt.title('Actual: %s\nPred: %s' % (actual_gender, pred_gender))
    plt.imshow(Image.open(ds_path+"/"+sample.filename))

plt.savefig(results_folder+"/example.jpg")