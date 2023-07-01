# Guess the age and the gender

This project aims to estimate in real-time the gender and an age-group of the person interacting with Pepper robot.

To do this, a dataset consistent with Pepper's webcam style must be available. [JoJoGAN](https://github.com/bortoletti-giorgia/JoJoGAN-Windows) and [SAM](https://github.com/bortoletti-giorgia/SAM) will be used for this purpose.

JoJoGAN applies the style of some reference images called "style references" to other images called "test inputs". In this project, "style references" are some pictures taken from Pepper's camera and "test input" is the UTKFace dataset.

After creating the dataset, models are trained.
The models are validated during an interaction experiment with Pepper and the data analysed to verify the performance of the model and perception and trust on the robot.

## Local Folder Structure

Below is the structure of the local folders:

```
.C:\0_thesis\
├── 0_dataset-analysis
│   ├── add-age-group.py			# Add age-group column in CSV files
│   ├── age_groups.py				# Definition of age-group
│   ├── analyse-adience.ipynb			# Analysis on age and gender distribution in Adience dataset
│   ├── analyse-imdbwiki.ipynb			# Analysis on age and gender distribution in IMDB-Wiki dataset
│   ├── analyse-utkface.ipynb			# Analysis on age and gender distribution in UTKFace Align&Crop dataset
│   ├── analyse-utkface-wild.ipynb		# Analysis on age and gender distribution in UTKFace into-the-wild dataset
│   ├── create-csv-adience.py			# Create CSV file from Adience dataset in DeepLake
│   ├── create-csv-imdbwiki.py			# Create CSV file from IMDB-Wiki dataset
│   ├── create-csv-utkface.py			# Create CSV file from UTKFace dataset
│   ├── create-dataset-folders.ipynb		# Create dataset in gender or age-group folders
│   ├── create-dataset-json.ipynb		# JSON for Stylegan2-ada-pytorch
│   ├── resize-images.py			# Resize images with power-of-two size
│   ├── main.ipynb

├── \dataset
│   ├── adience
│   ├── imdb_crop
│   ├── wiki_crop
│   ├── UTKFace
│   ├── UTKFace-wild
│
│   ├── adience-data.csv
│   ├── imdbwiki-data.csv
│   ├── utkface-data.csv
│   ├── utkface-wild-data.csv

├── \1_stylegan\

├── \jojogan-model\
│   ├── JoJoGAN					# Clone of https://github.com/bortoletti-giorgia/JoJoGAN-Windows
│   ├── ├── align-faces.py              
│   ├── ├── main-create-own-style.py
│   ├── ├── main.py
│
│   ├── from-rosbag-to-images.py
│   ├── main-jojo.job
│   ├── main-jojo-align.job
│
│   ├── \test-style-config\

├── \sam-model\
│   ├── SAM					# Clone of https://github.com/bortoletti-giorgia/SAM
│   ├── main-sam.job

├── \2_model\

├── \3_experiment\

├── \4_results\

├── LICENSE
└── README.md
```

## Coding environment

Environment to use JoJoGAN and SAM are explained in the dedicated repositories.

Environment to launch the other codes is:
* Windows 11
* Anaconda 
* An environment in Anaconda called "thesi" and created from [environment.yml](https://github.com/bortoletti-giorgia/facial-age-estimation/blob/main/environment.yml).

The heaviest codes are launched with the [SLURM CLUSTER DEI](https://www.dei.unipd.it/bladecluster). For interfacing, the WinSCP application with Putty is used.

## Prepare dataset

## Analyse the datasets and create needed files

The datasets considered are: UTKFace Align&Crop, UTKFace into-the-wild, IMDB-WIKI, Adience.

Some demographic analyses are under the folder "dataset-analysis".

For the purposes of this project, the age range to which a person belongs will be taken into account and not his/her exact age. To build the needed files, use step-by-step: [main.ipynb](https://github.com/bortoletti-giorgia/facial-age-estimation/blob/dataset/dataset-analysis/main.ipynb). It will:
* create a CSV for all dataset in the same folder where datasets are
* add age-group column as defined in [age_groups.py](https://github.com/bortoletti-giorgia/facial-age-estimation/blob/dataset/dataset-analysis/age_groups.py)

There are also optional steps, including: starting from the dataset create a folder of images divided into subfolders according to gender or age-group.

Our "test inputs" will be the dataset UTKFace Align&Crop.

## Apply Face Stylization

### Take face images with Pepper robot camera

Training the model requires that the images in the dataset agree with the images to be taken from the Pepper. To do this, the first step is to capture faces images from the Pepper camera. They will be out "style references".

To do this, the steps are the following.

Environment:
* Ubuntu 16.04
* Python 2.7
* Choreographe 2.5.10.7
* Ros Kinetic: http://wiki.ros.org/kinetic/Installation/Ubuntu 
* Ros naoqi-sdk and pynaoqi: https://wiki.ros.org/nao/Tutorials/Installation 
* Ros for Pepper: http://wiki.ros.org/pepper/Tutorial_kinetic  

Write on terminal: 
```echo 'export PYTHONPATH=/home/giorgia/naoqi/pynaoqi-python2.7-2.1.4.13-linux64/:/opt/ros/kinetic/lib/python2.7/dist-packages' >> ~/.bashrc```
```echo 'export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:~/naoqi/pynaoqi-python2.7-2.1.4.13-linux64' >> ~/.bashrc```
```echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:~/naoqi/naoqi-sdk-2.1.4.13-linux64/lib' >> ~/.bashrc```
Check if *naoqi* works, write: ```python``` and then ```import naoqi```. No errors means that the installation is correct.


I terminal: ```roscore```

II terminal: ```roslaunch pepper_bringup pepper_full_py.launch nao_ip:=<ip_pepper>```

III terminal: ```rviz```
	Fixed frame: odom
	add: by display type: robot model
		by topic: pepper_robot /camera /front /image

IV terminal: ```rosbag record /pepper_robot/camera/front/image_raw```
		ctrl + C for stopping

Show the captured frames in sequence (it works with roscore activated): ```rqt_bag <nome_bag>```

From ROSBAG to images: ```python from-rosbag-to-images.py --dir=""```

Now we want to extract only the images with a face. Face is extracted using the same method used in JoJoGAN (*e4e_projection*) ```python align-faces.py --inputdir='/home/bortoletti/rosabag/' --outputdir='/home/bortoletti/rosbag-aligned/'```. [align-faces.py](https://github.com/bortoletti-giorgia/facial-age-estimation/blob/jojogan/jojogan/align-faces.py) should be placed inside */JoJoGAN* that is the clone of the [JoJoGAN](https://github.com/bortoletti-giorgia/JoJoGAN-Windows) and the environment should be set as explained in the same repository. As JOB file, [main-jojo-align.job](https://github.com/bortoletti-giorgia/facial-age-estimation/blob/jojogan/jojogan/main-jojo-align.job) can be used and other code remains unchanged.

Faces images are saved and to use them in JoJoGAN they should be placed in the folder */JoJoGAN/style_images*.

### Balance the age distribution in style references: SAM

Before using JoJoGAN we need n images for each age group. Our images are taken from people between the ages of 20 and 30, so we want to generalise them to all other age groups.

We can use [SAM](https://github.com/yuval-alaluf/SAM) and in particular refer to my [fork](https://github.com/bortoletti-giorgia/SAM). Use SLURM-UNIPD with the job file [here](https://github.com/bortoletti-giorgia/facial-age-estimation/blob/sam/sam/main-sam.job). The target ages are defined as the middle ages of each age-group via the [getMiddles()](https://github.com/bortoletti-giorgia/facial-age-estimation/blob/dataset/dataset-analysis/age_groups.py) method so that in the subsequent style adaptation the style reference image are equidistant from the minimum and maximum of each range.

In *SAM/notebooks/images* place the selected style references images to generalise (some obtained from the ROSBAGS). Results will be saved in *SAM/results*.

### Apply style to dataset: JoJoGAN

#### SLURM Workspace Structure

Code is heavy and it will be launch in SLURM.

Your workspace structure should be (“bortoletti” is the example workspace):

```
\home\bortoletti
├── JoJoGAN                           # clone of https://github.com/bortoletti-giorgia/JoJoGAN-Windows
├── ├── inversion_codes               # folder created after first execution
├── ├── style_images                  # folder created after first execution
├── ├── style_images_aligned          # folder created after first execution 
├── ├── models                        # folder created after first execution
├── ├── results                       # folder created after first execution 
├── ├──  main.py                      # main code to run JoJoGAN with pretrained model
├── ├──  main-create-own-style.py     # main code to create a model with your style images 
├── ├──  align-faces.py               # optional code that align and crop some face images

├── out                               # folder with TXT file with errors and shell output of main.job 
│   main.job                          # JOB file for running JoJoGAN 
│   main-align.job                    # JOB file for running align-faces.py using JoJoGAN 
│   singularity-container.sif         # Singularity container for executing the JOB file
```

#### How many style images to use? And which?

"Style references" should be a combination of original images taken from Pepper and those obtained from SAM.

To choose the number and the configuration we made some tests with: 1 image, 5 images and 6 images. 

For 6 images, steps are the following:
1. Create a small dataset of 3 images from UTKFace Align&Crop with ```python create-dataset-folders.py --outdirgender='C:/0_thesis/dataset/utkface-small-ranges' --source='C:/0_thesis/dataset/' --filename='utkface-data.csv' --nsamples=3```
2. Copy *'C:/0_thesis/dataset/utkface-small-ranges'* to SLURM in *'/home/bortoletti/JoJoGAN/test_input/utk-test'*
3. Choose 6 images from the ones obtained from ROSBAGS and copy them in *'C:\0_thesis\1_stylegan\sam-model\results-6shots\ori'* and *'/home/bortoletti/SAM/notebooks/images'*
4. Launch SAM as above
5. Copy *'/home/bortoletti/SAM/results* in *'C:\0_thesis\1_stylegan\sam-model\results-6shots'*
6. Create different configurations with [create-test-style-folders](https://github.com/bortoletti-giorgia/facial-age-estimation/blob/jojogan/jojogan/create-test-style-folders.py) (check that the paths it accesses are consistent with yours)
7. Copy "style references" from *'C:\0_thesis\1_stylegan\jojogan-model\test-style-config\style_images-6shots'* to *'/home/bortoletti/JoJoGAN/style_images'*
8. Launch the JOB [main-jojo-config.job](https://github.com/bortoletti-giorgia/facial-age-estimation/blob/jojogan/jojogan/main-jojo-config.job)

Configurations are:
* Configuration 0: 1 original image + 5 obtained from SAM
* Configuration 1: 2 original images + 4 obtained from SAM
* Configuration 2: 3 original images + 3 obtained from SAM
* Configuration 3: 4 original images + 2 obtained from SAM
* Configuration 4: 5 original images + 1 obtained from SAM

Models name are composed as "pepper_"+*middle age of each age-group*+"-"+*config id from 0 to 4*.

There are not relevant differences, so we proceed with 6 shots and configuration 1 that is the most balanced for age and gender.

#### Create Pepper dataset

Final step is to run JoJoGAN with UTKFace in "test_input" divided in age-groups and 6 shots with config. 1 in "style_images"

JOB file is [main-jojo.job](https://github.com/bortoletti-giorgia/facial-age-estimation/blob/jojogan/jojogan/main-jojo.job).

## Train
Training is done on gender and age labels. However, age is not considered in its exactness but taking into account an age-group defined in [age_groups.py](https://github.com/bortoletti-giorgia/facial-age-estimation/blob/dataset/dataset-analysis/age_groups.py). 

All training steps are enumerated in [this branch](https://github.com/bortoletti-giorgia/facial-age-and-gender-estimation/tree/model/model). The final training model is [for RGB images](https://github.com/bortoletti-giorgia/facial-age-and-gender-estimation/blob/model/model/2_train-cnn-model-fold.py) and [for grayscale images](https://github.com/bortoletti-giorgia/facial-age-and-gender-estimation/blob/model/model/2_train-cnn-model-fold-gray.py).

## Experiment

All the code is in [this branch](https://github.com/bortoletti-giorgia/facial-age-and-gender-estimation/tree/experiment/experiment) and explanation is written in the thesis documentation.

## Evaluation

Experiment data are elaborated with code [here](https://github.com/bortoletti-giorgia/facial-age-and-gender-estimation/tree/evaluation/evaluation) and final consideration reported in the thesis documentation.


 
