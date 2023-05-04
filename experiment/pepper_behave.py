import qi
import time
import sys
import argparse
import random
from age_groups import *


class AgeGroupBehave(object):

	def __init__(self, app, age, gender):
		"""
		Initialisation of qi framework and event detection.
		"""
		super(AgeGroupBehave, self).__init__()
		# Create folder where to save captured images
		self.age_group = AgeGroups().getGroupFromAge(age)
		self.gender = gender

		app.start()
		session = app.session
		# Get some services
		self.memory = session.service("ALMemory")
		self.leds_service = session.service("ALLeds")
		self.tts = session.service("ALTextToSpeech")
		self.animated_speech = session.service("ALAnimatedSpeech")
		self.tablet_service = session.service("ALTabletService")
		self.photo_service = session.service("ALPhotoCapture")

		#self.tts.setLanguage("Italian")
		
    #-------------------------------------------------------------------------------------------
	def test_behave(self):
		
		# Farlo stare fermo da quando intercetta una persona??
		print("Test animations.")
		
		# Show image on Tablet
		image_path = "/path/to/image.jpg"
		image = self.photo_service.getImageLocal(image_path)
		self.tablet_service.showImage(image)

		# Led and animation during dialog
		duration = 1.0
		rgb_color = [255, 0, 0]
		self.leds_service.fadeRGB("FaceLeds", rgb_color, duration)
		self.leds_service.fadeRGB("FaceLeds", rgb_color, duration)

		animation = "EyesAnim/Basic/Neutral/SoftBank"
		text = "Hello world!"
		self.tts.say("^start(%s) %s ^wait(%s)" % (animation, text, animation))
		self.leds_service.reset("FaceLeds")

		# Pepper default tags: http://doc.aldebaran.com/2-5/naoqi/motion/alanimationplayer-advanced.html#animationplayer-list-behaviors-pepper
		tag = "me"
		text = "Hello, I'm Pepper. Nice to meet you!"
		self.animated_speech.say("^start(%s) %s ^wait(%s)" % (tag, text, tag))

		tag = "explain"
		self.animated_speech.say("^startTag(explain)Hello! I am Pepper robot.^stopTag(present)")

		# Define the animation name, eye color and color duration 
		animation_name = "EyesAnim/Hello"
		eye_color = (255, 0, 0)  # Red
		duration = 0.5
		# Define the text to be spoken
		text = "Hello world!"
		# Set the eye color and start the animation
		self.leds_service.fadeRGB("FaceLeds", eye_color, duration)
		self.animated_speech.say("^start(%s) %s ^wait(%s)" % (animation_name, text, animation_name))
		# Reset the eye color after the animation has finished
		time.sleep(2)  # Wait for the animation to finish
		self.leds_service.fadeRGB("FaceLeds", (0, 0, 0), duration)
	
	#-------------------------------------------------------------------------------------------
	def behave(self):
		# 0-2 or 3-12
		if self.age_group == 0 or self.age_group == 1: 
			print("Under 18: bad.")
		# 18-29 because I know that people coming in the lab are over 18
		elif self.age_group == 2 or self.age_group == 3: 
			print("18-29 years")
			self.case_3()
		# 30-39 years
		elif self.age_group == 4: 
			print("30-39 years")
			self.case_4()
		# 40-49 years
		elif self.age_group == 5: 
			print("40-49 years")
			self.case_5()
		# 50-59 years
		elif self.age_group == 6: 
			print("50-59 years")
			self.case_6()
		# 60-69 years
		elif self.age_group == 7: 
			print("60-69 years")
			self.case_7()
		# over 70
		else: 
			print("Over 70 years")
			self.case_8()

	#-------------------------------------------------------------------------------------------
	def case_3(self): # 18-29 years
		None

	def case_4(self): # 30-39 years
		None

	def case_5(self): # 40-49 years
		None

	def case_6(self): # 50-59 years
		None

	def case_7(self): # 60-69 years
		None

	def case_8(self): # over 70
		None

#-------------------------------------------------------------------------------------------
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--ip", type=str, default="172.20.10.10", help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
	parser.add_argument("--port", type=int, default=9559, help="Naoqi port number")
	parser.add_argument("--age", help="estimated person age", required=True)
	parser.add_argument("--gender", help="estimated person gender", required=True)

	args = parser.parse_args()
	try:
		# Initialize qi framework.
		connection_url = "tcp://" + args.ip + ":" + str(args.port)
		app = qi.Application(["AgeGroupBehave", "--qi-url=" + connection_url])
	except RuntimeError:
		print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
		"Please check your script arguments. Run with -h option for help.")
		sys.exit(1)

	human_greeter = AgeGroupBehave(app, age=args.age, gender=args.gender)
	human_greeter.behave()
