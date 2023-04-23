from naoqi import ALProxy
import numpy as np
import cv2
import argparse
import os
import time

THRESHOLD_BLURRY = 70

def is_valid_image(img):
	# Load the cascade
	# https://github.com/opencv/opencv/tree/master/data/haarcascades
	face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
	# Convert into grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Detect faces
	faces = face_cascade.detectMultiScale(
	    gray,
	    scaleFactor=1.1,
	    minNeighbors=5,
	    minSize=(30, 30)
	)
	# Draw rectangle around the faces
	if len(faces) == 0:
		return False
	for (x, y, w, h) in faces:
		face = img[y:y+h, x:x+w]
		# check if image is blurry
		gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
		fm = cv2.Laplacian(gray, cv2.CV_64F).var()
		# if the focus measure is less than the supplied threshold,
		# then the image should be considered "blurry"
		blurry = True if fm<THRESHOLD_BLURRY else False
		if blurry:
			return False
		else:
			#cv2.imshow("valid", gray)
			#cv2.waitKey(0)
			return True
			
	return False

#-------------------------------------------------------------------------------------------
# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--ip", help="IP of Pepper robot", type=str, default="192.168.0.106")
parser.add_argument("--port", help="PORT of Pepper robot", type=int, default=9559)
parser.add_argument("--outputdir", help="directory where captured images will be saved", required=True)

args = parser.parse_args()

IP = args.ip
PORT = args.port
output_dir = args.outputdir

if not os.path.exists(output_dir):
	os.mkdir(output_dir)

# Say something
#tts = ALProxy("ALTextToSpeech", IP, PORT)

#tts.say("Hello, I am collecting some photos.")

# Capture photo
# https://gist.github.com/takamin/990aa0133919aa58944d

video_device = ALProxy('ALVideoDevice', IP, PORT)
awareness = ALProxy("ALBasicAwareness", IP, PORT)
awareness.setEnabled(False)

# Subscribe top camera
# https://fileadmin.cs.lth.se/robot/nao/doc/family/juliette_technical/video_juliette.html#juliette-video
AL_kTopCamera = 0
AL_kQVGA = 2 #1: 320x240; 2: 640x480
AL_kBGRColorSpace = 13
fps = 1
capture_device = video_device.subscribeCamera("test", AL_kTopCamera, AL_kQVGA, AL_kBGRColorSpace, fps)

# create image
#width = 320*AL_kQVGA
#height = 240*AL_kQVGA
width = 640
height = 480
image = np.zeros((height, width, 3), np.uint8)

# get image
k = 0
while k<5:
	result = video_device.getImageRemote(capture_device)

	if result == None:
		print('cannot capture.')
	elif result[6] == None:
		print('no image data string.')
	else:
		# translate result to mat
		values = map(ord, list(result[6]))
		i = 0
		for y in range(0, height):
			for x in range(0, width):
				image.itemset((y, x, 0), values[i + 0])
				image.itemset((y, x, 1), values[i + 1])
				image.itemset((y, x, 2), values[i + 2])
				i += 3
		if is_valid_image(image):
			print("Image "+str(k)+" saved.")
			cv2.imwrite(output_dir+"/rgb_"+str(k)+".jpg", image)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			cv2.imwrite(output_dir+"/grayscale_"+str(k)+".jpg", gray)
			k += 1
	#time.sleep(0.1)


#tts.say("I've done.")
