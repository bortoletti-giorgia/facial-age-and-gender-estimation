
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys
import argparse
import time
import cv2
from pepper_greeter import *
from pepper_behave import *
import multiprocessing
import signal

class Experiment:
	root = "/home/giorgia/0_thesis/3_experiment/"
	#root = "C:/0_thesis/3_experiment/"

	#-------------------------------------------------------------------------------------------
	def __init__(self, id_exp, colormode="rgb"):
		"""
		Initialisation of Experiment.
		"""
		self.exp_folder = os.path.join(self.root, "results/"+str(id_exp)+"/")
		if not os.path.exists(self.exp_folder):
			os.mkdir(self.exp_folder)

		# Cropped and aligned images 
		self.img_folder = os.path.join(self.exp_folder, "photos/")
		if not os.path.exists(self.img_folder):
			os.mkdir(self.img_folder)

		# Valid captured images
		self.temp_img_folder =  os.path.join(self.exp_folder, "temp_photos/")
		if not os.path.exists(self.temp_img_folder):
			os.mkdir(self.temp_img_folder)

		# Not valid captured images -> for future experiments
		self.trash_img_folder =  os.path.join(self.exp_folder, "trash_photos/")
		if not os.path.exists(self.trash_img_folder):
			os.mkdir(self.trash_img_folder)

		self.colormode = colormode
		if colormode == "rgb":
			self.model_path = os.path.join(self.root, "models/rgb_no_alpha/model_4")
		elif colormode == "grayscale":
			self.model_path = os.path.join(self.root, "models/gray_no_alpha/model_4")

		# Prediction output TXT file
		self.prediction_file = self.exp_folder+"/prediction_"+self.colormode+".txt"

	#-------------------------------------------------------------------------------------------
	def init_robot(self, ip, port, image_folder, not_valid_image_folder):
		"""
		Create two robot sessions: HumanGreeter and AgeGroupBehave.
		Two different instances because each has specific services.
		Return True if connection is possible, False otherwise.
		"""
		# Connect to the robot
		try:
			# Initialize qi framework.
			connection_url = "tcp://" + str(ip) + ":" + str(port)
			app = qi.Application(["HumanGreeter", "--qi-url=" + connection_url])
		except RuntimeError:
			print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
			"Please check your script arguments. Run with -h option for help.")
			return False

		# Two different instances because each has specific services.
		self.robot_greeter = HumanGreeter(app=app, image_folder=image_folder, not_valid_image_folder=not_valid_image_folder)
		self.robot_behave = AgeGroupBehave(app=app)
		return True

	#-------------------------------------------------------------------------------------------
	def capture_images(self, ip, port):		
		"""
		Calling the file 'pepper_greeter.py', capture photos from Pepper's camera.
		Pepper does NOT present itself.
		"""
		# When a person starts looking at it, it stops and capture images
		self.ip_robot = str(ip)
		self.port_robot = str(port)
		cmd_capture_images = "python pepper_greeter.py --ip='"+str(ip)+"' --port="+str(port)+" --outputdir='"+self.temp_img_folder+"'"
		# Or it is blocked since beginning and capture images
		#cmd_capture_images = "python pepper_capture_images.py --ip='"+str(ip)+"' --port="+str(port)+" --outputdir='"+self.temp_img_folder+"'"
		os.system(cmd_capture_images)

	#-------------------------------------------------------------------------------------------
	def crop_image_haarcascade(self):
		"""
		Align and crop images on face according to 'haarcascade_frontalface_default.xml'.
		"""
		print("Start to align and crop images with HAARCASCADE.")
		inputdir = self.temp_img_folder
		outputdir = self.img_folder
		files = os.listdir(inputdir)
		for filename in files:
			if filename[-4:] == '.jpg': # image
				img = cv2.imread(inputdir+filename)
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
					fm = cv2.Laplacian(gray, cv2.CV_64F).var()
					# if the focus measure is less than the supplied threshold,
					# then the image should be considered "blurry"
					blurry = True if fm<30 else False
					if not blurry:
						gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
						cv2.imwrite(outputdir+"/rgb_"+filename, face)
						cv2.imwrite(outputdir+"/grayscale_"+filename, gray)
						#cv2.imshow(" ("+str(fm)+")", gray)
						#cv2.waitKey(0)
		cv2.destroyAllWindows() 

	#-------------------------------------------------------------------------------------------
	def crop_images_jojogan(self):
		"""
		Activate the Anaconda environment 'jojo' and 
		align and crop images on face according to JoJoGAN alignment. The newly cropped images are used as input for prediction.
		"""
		print("Start to align and crop images with JOJOGAN.")
		# Activate the "jojo" Anaconda env
		# Enter in JoJoGAN folder and run align-faces.py script
		cmd_align_faces = 'cd JoJoGAN && conda run -n jojo python align-faces.py --inputdir="'+self.temp_img_folder+'" --outputdir="'+self.img_folder+'"'
		os.system(cmd_align_faces)

	#-------------------------------------------------------------------------------------------
	def predict(self):
		"""
		Activate the Anaconda environment 'model' and 
		return an age and gender prediction in a TXT file inside the experiment main folder.
		TXT file structure: <age>,<gender>. Age is an integer. Gender is "female" or "male".
		"""
		# Activate the "model" Anaconda env and run predict.py script
		os.system('conda run -n model python predict.py --imagespath="'+self.img_folder+'" --resultfile="'+self.prediction_file+'" --modelpath="'+self.model_path+'" --colormode="'+self.colormode+'"')
	
	#-------------------------------------------------------------------------------------------
	def get_age(self):
		"""
		Return age of the person associated in this experiment.
		"""
		return self.age
	def get_gender(self):
		"""
		Return gender of the person associated in this experiment.
		"""
		return self.gender
	
	#-------------------------------------------------------------------------------------------
	def predict_all_models(self):
		"""
		Use the predict fuction on all the availables models.
		"""
		models_rgb = []
		models_rgb.append(os.path.join(self.root, "models/rgb_no_alpha/model_4"))
		models_rgb.append(os.path.join(self.root, "models/rgb_alpha05/model_4"))
		models_rgb.append(os.path.join(self.root, "models/rgb_alpha1/model_4"))

		models_gray = []
		models_gray.append(os.path.join(self.root, "models/gray_no_alpha/model_4"))
		models_gray.append(os.path.join(self.root, "models/gray_alpha05/model_4"))
		models_gray.append(os.path.join(self.root, "models/gray_alpha1/model_4"))
		
		#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
		for model in models_rgb:
			print(model)
			result_file = self.exp_folder+"/prediction_"+model.split("/")[-2]+".txt"
			os.system('conda run -n model python predict.py --imagespath="'+self.img_folder+'" --resultfile="'+result_file+'" --modelpath="'+model+'" --colormode="rgb"')

		for model in models_gray:
			print(model)
			result_file = self.exp_folder+"/prediction_"+model.split("/")[-2]+".txt"
			os.system('conda run -n model python predict.py --imagespath="'+self.img_folder+'" --resultfile="'+result_file+'" --modelpath="'+model+'" --colormode="grayscale"')
	
	#-------------------------------------------------------------------------------------------
	def predict_balance_models(self):
		"""
		Use the predict fuction on all the availables balanced models.
		"""
		temp_root = "C:/0_thesis/2_model/TESTING/BALANCE/"

		models_rgb = []
		models_rgb.append(os.path.join(temp_root, "19fold/model_4"))
		models_rgb.append(os.path.join(temp_root, "20netto/model_4"))
		models_rgb.append(os.path.join(temp_root, "20pepper05/model_4"))

		models_gray = []
		models_gray.append(os.path.join(temp_root, "19foldgray/model_4"))
		models_gray.append(os.path.join(temp_root, "20nettogray/model_4"))
		models_gray.append(os.path.join(temp_root, "20pepper05gray/model_4"))
		
		#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
		for model in models_rgb:
			print(model)
			result_file = self.exp_folder+"/prediction_balance_"+model.split("/")[-2]+".txt"
			os.system('conda run -n thesis python predict.py --imagespath="'+self.img_folder+'" --resultfile="'+result_file+'" --modelpath="'+model+'" --colormode="rgb"')

		for model in models_gray:
			print(model)
			result_file = self.exp_folder+"/prediction_balance_"+model.split("/")[-2]+".txt"
			os.system('conda run -n thesis python predict.py --imagespath="'+self.img_folder+'" --resultfile="'+result_file+'" --modelpath="'+model+'" --colormode="grayscale"')

	#-------------------------------------------------------------------------------------------
	def behave(self):
		"""
		Calling the file 'pepper_behave.py', reproduce a customized behavior on Pepper based on experiment's age and gender.
		"""
		age = self.get_age()
		gender = self.get_gender()
		cmd_behave = "python pepper_behave.py --ip='"+self.ip_robot+"' --port="+self.port_robot+" --age="+age+" --gender="+gender
		os.system(cmd_behave)

	#-------------------------------------------------------------------------------------------
	def run(self):	
		"""
		Run the experiment. Capture face images and crop them while the robot presents itself.
		Predict age and gender. Do some customized behaviors based on the prediction.
		"""	
		# Init robot
		if not self.init_robot(ip=ip_robot, port=port_robot, image_folder=self.temp_img_folder, not_valid_image_folder=self.trash_img_folder):
			sys.exit(1)

		# Capture images
		self.robot_greeter.run()

		# At this point capturing is done -> 
		# In PARALLEL:
		# 1: Crop images on faces, then predict age and gender
		# 2: Pepper introduces itself
		process_crop_images = multiprocessing.Process(target=self.crop_images_jojogan)
		process_intro_robot = multiprocessing.Process(target=self.robot_greeter.introduce_robot)
		process_predict = multiprocessing.Process(target=self.predict)
		
		process_intro_robot.start()
		process_crop_images.start()
		#process_intro_robot.join()
		process_crop_images.join()
		
		if not process_crop_images.is_alive():
			process_crop_images.terminate()
			process_predict.start()
			process_predict.join()

		if not process_predict.is_alive():	
			# PREDICTION
			# At this point cropping is done -> Proceed with prediction
			#self.predict()
			# Read the TXT prediction file
			# and save age and gender in the experiment object
			f = open(self.prediction_file, "r")
			content = f.read()
			self.age = content.split(",")[0]
			self.gender = content.split(",")[1]
			f.close()
			print("Final age: ", self.age)
			print("Final gender: ", self.gender)
			# DO BEHAVIORS
			#self.robot_behave.behave(age=self.age, gender=self.gender)	

#-------------------------------------------------------------------------------------------
if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--ip", type=str, default="172.20.10.10", help="Robot ip address.")
	parser.add_argument("--port", type=int, default=9559, help="Robot port number.")
	parser.add_argument("--colormode", help="rgb or grayscale", type=str, default="grayscale")

	args = parser.parse_args()
	ip_robot = args.ip
	port_robot = args.port

	# START
	start_time = time.time()
	# Init experiment
	id_exp = 24
	print("Experiment: ", str(id_exp))
	exp = Experiment(id_exp, colormode=args.colormode)
	# Run
	exp.run()
	# END
	print("--- %s seconds ---" % round(time.time() - start_time, 2))

	''' ANALYSES ALL EXPERIMENTS
	subfolders = os.listdir("C:/0_thesis/3_experiment/results/")
	for s in subfolders:
		id_exp = int(s)
		print("Experiment: ", s)
		exp = Experiment(id_exp, colormode=args.colormode)
		# PREDICTION
		#exp.predict_all_models()
		#exp.predict_balance_models()
	'''

