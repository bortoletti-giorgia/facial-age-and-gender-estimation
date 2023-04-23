
#http://doc.aldebaran.com/2-5/dev/python/examples/vision/face_detection.html

import qi
import time
import sys
import argparse
import numpy as np
import os
import cv2

THRESHOLD_BLURRY = 70

class HumanGreeter(object):
	"""
	A simple class to react to face detection events.
	"""

	def __init__(self, app, outputdir):
		"""
		Initialisation of qi framework and event detection.
		"""
		super(HumanGreeter, self).__init__()
		# Create folder where to save captured images
		self.output_dir = outputdir
		if not os.path.exists(self.output_dir):
			os.mkdir(self.output_dir)
		
		app.start()
		session = app.session
		# Get the service ALMemory.
		self.memory = session.service("ALMemory")
		# Connect the event callback.
		self.subscriber_gaze = self.memory.subscriber("GazeAnalysis/PersonStartsLookingAtRobot")
		self.subscriber_gaze.signal.connect(self.on_human_tracked)
		self.subscriber_face = self.memory.subscriber("FaceDetected")
		#self.people_perception = session.service("ALTracker")
		self.subscriber_people = self.memory.subscriber("PeoplePerception/PeopleDetected")
		# Get the services ALTextToSpeech, ALFaceDetection, ALPeoplePerception,
		# ALRobotPosture, ALBasicAwareness, ALVideoDevice
		self.posture = session.service("ALRobotPosture")
		self.tts = session.service("ALTextToSpeech")
		self.face_detection = session.service("ALFaceDetection")
		self.people_perception = session.service("ALPeoplePerception")
		self.awareness = session.service("ALBasicAwareness")
		self.video_device = session.service("ALVideoDevice")

		self.face_detection.subscribe("HumanGreeter")
		self.awareness.setEnabled(True) # Robot is enabled for catching a person
		self.got_face = False
		
    #-------------------------------------------------------------------------------------------
	def on_human_tracked(self, value):
		"""
		Callback for event GazeAnalysis/PersonStartsLookingAtRobot.
		"""
		if value == []:  # empty value when the face disappears
			self.got_face = False
		elif not self.got_face:
			self.got_face = True
			# GAZE DETECTION
			self.awareness.setEnabled(False) # Disabled person detection
			print("Someone is looking at me.")
			self.tts.say("Hello, you!")
			value_face = self.memory.getData("FaceDetected", 0)

			# PEOPLE PERCEPTION
			value_person = self.memory.getData("PeoplePerception/PeopleDetected", 0)
			timeStamp = value_person[0]
			print("TimeStamp is: " + str(timeStamp))
			personInfoArray = value_person[1][0]
			person_id = personInfoArray[0]
			distanceToCamera = personInfoArray[1]
			pitchAngleInImage = personInfoArray[2]
			yawAngleInImage = personInfoArray[3]
			print("Person Info: ID %.3f - distance %.3f" % (person_id, distanceToCamera))
			print("Person In Image (radians):  pitch %.3f - yaw %.3f" % (pitchAngleInImage, yawAngleInImage))
			
			# Head Pepper coordinates
			#yaw, pitch, roll = self.memory.getData("PeoplePerception/Person/"+str(person_id)+"/HeadAngles")
			#print "Head Robot (degrees):  --yam=%.3f --pitch=%.3f --roll=%.3f" % (np.rad2deg(yaw), np.rad2deg(pitch), np.rad2deg(roll))

			self.tts.say("Stop here.")
			#self.posture.stopMove()
			self.capture_images()
			#time.sleep(10.0)
			self.awareness.setEnabled(True) # Enabled person detection
			self.tts.say("Ready.")
			#self.got_face = False

			# FACE DETECTION
			'''
			print "I saw a face!"
			#self.tts.say("Nice to meet you.")
			# First Field = TimeStamp.
			timeStamp = value_face[0]
			print "TimeStamp is: " + str(timeStamp)
			# Second Field = array of face_Info's.
			faceInfoArray = value_face[1]
			for j in range( len(faceInfoArray)-1 ):
				faceInfo = faceInfoArray[j]

				# First Field = Shape info.
				faceShapeInfo = faceInfo[0]

				# Second Field = Extra info (empty for now).
				faceExtraInfo = faceInfo[1]

				print "Face Infos :  alpha %.3f - beta %.3f" % (faceShapeInfo[1], faceShapeInfo[2])
				print "Face Infos :  width %.3f - height %.3f" % (faceShapeInfo[3], faceShapeInfo[4])
				print "Face Extra Infos :" + str(faceExtraInfo)
			'''

	#-------------------------------------------------------------------------------------------
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
	def capture_images(self):
		# Subscribe top camera
		# https://fileadmin.cs.lth.se/robot/nao/doc/family/juliette_technical/video_juliette.html#juliette-video
		AL_kTopCamera = 0
		AL_kQVGA = 2 #1: 320x240; 2: 640x480
		AL_kBGRColorSpace = 13
		fps = 13
		subscriberID = self.video_device.subscribeCamera("test", AL_kTopCamera, AL_kQVGA, AL_kBGRColorSpace, fps)

		# create image
		width = 320*AL_kQVGA
		height = 240*AL_kQVGA
		image = np.zeros((height, width, 3), np.uint8)

		# get image
		k = 0
		while k < 5:
			result = self.video_device.getImageRemote(subscriberID)
			
			if result == None:
				print('cannot capture.')
			elif result[6] == None:
				print('no image data string.')
			else: # translate result to mat
				print(len(result[6]))
				values = list(result[6])
				i = 0
				for y in range(0, height):
					for x in range(0, width):
						image.itemset((y, x, 0), values[i + 0])
						image.itemset((y, x, 1), values[i + 1])
						image.itemset((y, x, 2), values[i + 2])
						i += 3
				# save image
				if self.is_valid_image(image):
					cv2.imwrite(self.output_dir+"/rgb_"+str(k)+".jpg", image)
					gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
					cv2.imwrite(self.output_dir+"/grayscale_"+str(k)+".jpg", gray)
					k += 1
					#time.sleep(0.1)
		self.video_device.unsubscribe(subscriberID)

	#-------------------------------------------------------------------------------------------
	def run(self):
		"""
		Loop on, wait for events until manual interruption.
		"""
		print("Starting HumanGreeter")
		try:
			while True:
				time.sleep(1)
		except KeyboardInterrupt:
			print("Interrupted by user, stopping HumanGreeter")
			self.face_detection.unsubscribe("HumanGreeter")
			#stop
			sys.exit(0)

#-------------------------------------------------------------------------------------------
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--ip", type=str, default="172.20.10.10", help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
	parser.add_argument("--port", type=int, default=9559, help="Naoqi port number")
	parser.add_argument("--outputdir", help="directory where captured images will be saved", required=True)

	args = parser.parse_args()
	try:
		# Initialize qi framework.
		connection_url = "tcp://" + args.ip + ":" + str(args.port)
		app = qi.Application(["HumanGreeter", "--qi-url=" + connection_url])
	except RuntimeError:
		print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
		"Please check your script arguments. Run with -h option for help.")
		sys.exit(1)

	human_greeter = HumanGreeter(app, args.outputdir)
	human_greeter.run()
