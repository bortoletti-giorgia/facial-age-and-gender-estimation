# Guess the age

This project aims to estimate in real-time the gender and an age-group of the person interacting with Pepper robot.

## Repository Structure
```
.
├── dataset-analysis
│   ├── analyse-adience.ipynb           # Analysis on age and gender distribution in Adience dataset
│   ├── analyse-imdbwiki.ipynb          # Analysis on age and gender distribution in IMDB-WIKI dataset
│   ├── analyse-utkface.ipynb           # Analysis on age and gender distribution in UTKFace dataset
│   ├── create-csv-adience.py           # Create CSV file from Adience dataset in DeepLake
│   ├── create-csv-imdbwiki.py          # Create CSV file from local IMDB-WIKI dataset
│   ├── create-csv-utkface.py           # Create CSV file from local UTKFace dataset
│   ├── age_groups.py                   # Definition of ranges to classify ages
│   ├── add-age-group.py                # Add 'age-group' column in each CSV file
│   ├── resize-images.py                # Resize images for StyleGan2-ADA-Pytorch
│   ├── main.ipynb                      # Analysis on all datasets
│   └── unit                
├── jojogan 
│   ├── pepper-style
│   ├── ├── from-rosbag-to-images.py    # Extract images from ROSABAG files
│   ├── ├── align-faces.py              # Save only images that contain a face

├── LICENSE
└── README.md
```
## Prepare dataset

For training datasets are: UTKFace, IMDB-WIKI, Adience.

Some demographic analyses are under the folder "dataset-analysis".
Use [main.ipynb](https://github.com/bortoletti-giorgia/facial-age-estimation/blob/dataset/dataset-analysis/main.ipynb) to create CSV for all datasets and add age groups for training.

## Apply Face Stylization

### Take face images with Pepper robot camera

Training the model requires that the images in the dataset agree with the images to be taken from the Pepper. To do this, the first step is to capture faces images from the Pepper camera and then apply their style to the chosen dataset (UTKFace etc.).
To do this, the steps are the following.

Environment:
* Ubuntu 16.04
* Choreographe 2.5.10.7
* Ros Kinetic: http://wiki.ros.org/kinetic/Installation/Ubuntu 
* Ros for Pepper: ```sudo apt-get install ros-kinetic-pepper-robot ros-kinetic-pepper-meshes```


I terminal: ```roscore```

II terminal: ```roslaunch pepper_bringup pepper_full_py.launch nao_ip:=192.168.0.106```

III terminal: ```rviz```
	Fixed frame: odom
	add: by display type: robot model
		by topic: pepper_robot /camera /front /image

IV terminal: ```rosbag record /pepper_robot/camera/front/image_raw```
		ctrl + C for stopping

Show the captured frames in sequence (it works with roscore activated): ```rqt_bag <nome_bag>```

From ROSBAG to images: ```python from-rosbag-to-images.py --dir=""```

Now we want to extract only the images with a face. Face is extracted using the same method used in JoJoGAN (*e4e_projection*) ```python align-faces.py```. *align-images.py* should be placed inside */JoJoGAN* that is the clone of the [JoJoGAN](https://github.com/bortoletti-giorgia/JoJoGAN-Windows) and the environment should be set as explained in the same repository.

Faces images are saved and to use them in JoJoGAN they should be placed in the folder */JoJoGAN/style_images*.

### Balance the age distribution in the dataset: SAM

Before using JoJoGAN you need to have n images for each age-group. In our case 4 images for each range. This is because once you go to apply the style on an image you need the age-group to be consistent and not, for example, apply the style of a 20 year old boy on an 80 year old person. You can use [SAM](https://github.com/yuval-alaluf/SAM) to do this. To use SAM refer to my [fork](https://github.com/bortoletti-giorgia/SAM). Use SLURM-UNIPD with the job file [here](https://github.com/bortoletti-giorgia/facial-age-estimation/blob/sam/sam/main-sam.job). The target ages are defined as the middle ages of each age-group via the [getMiddles()](https://github.com/bortoletti-giorgia/facial-age-estimation/blob/dataset/dataset-analysis/age_groups.py) method so that in the subsequent style adaptation the style reference image are equidistant from the minimum and maximum of each range.


### Apply style to dataset: JoJoGAN


## Train
Training is done on gender and age labels. However, age is not considered in its exactness but taking into account an age-group defined in [age_groups.py](https://github.com/bortoletti-giorgia/facial-age-estimation/blob/dataset/dataset-analysis/age_groups.py). 

## Test

## Evaluate
 
