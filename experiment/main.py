import os
import sys
import argparse
import time
from age_groups import *
import cv2

class Experiment:
	#root = "/home/giorgia/0_thesis/3_experiment/"
	root = "C:/0_thesis/3_experiment/"

	#-------------------------------------------------------------------------------------------
	def __init__(self, id_exp, colormode="rgb"):
		self.exp_folder = os.path.join(self.root, "results/"+str(id_exp)+"/")
		if not os.path.exists(self.exp_folder):
			os.mkdir(self.exp_folder)

		self.img_folder = os.path.join(self.exp_folder, "photos/")
		if not os.path.exists(self.img_folder):
			os.mkdir(self.img_folder)

		self.temp_img_folder =  os.path.join(self.exp_folder, "temp_photos/")
		if not os.path.exists(self.temp_img_folder):
			os.mkdir(self.temp_img_folder)

		self.colormode = colormode
		if colormode == "rgb":
			self.model_path = os.path.join(self.root, "models/19fold/model_4")
		elif colormode == "grayscale":
			self.model_path = os.path.join(self.root, "models/19foldgray/model_4")

	#-------------------------------------------------------------------------------------------
	def capture_images(self, ip, port):		
		# Capture photos and align on faces from Pepper's camera
		# When a person starts looking at it, it stops and capture images
		cmd_capture_images = "python pepper_greeter.py --ip='"+str(ip)+"' --port="+str(port)+" --outputdir='"+self.temp_img_folder+"'"
		# Or it is blocked since beginning and capture images
		#cmd_capture_images = "python pepper_capture_images.py --ip='"+str(ip)+"' --port="+str(port)+" --outputdir='"+self.temp_img_folder+"'"
		os.system(cmd_capture_images)

	#-------------------------------------------------------------------------------------------
	def align_faces(self, inputdir, outputdir):
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
						cv2.imshow(" ("+str(fm)+")", gray)
						cv2.waitKey(0)
		cv2.destroyAllWindows() 

	#-------------------------------------------------------------------------------------------
	def align_images(self):
		# Crop face from the image and save it in outputdir
		print("Start to align and crop images.")
		self.align_faces(inputdir=self.temp_img_folder, outputdir=self.img_folder)

	def align_images_jojogan(self):
		# Crop face from the image and save it in outputdir with JoJoGAN cutting
		print("Start alignment and cropping images.")
		cmd_align_faces = 'cd JoJoGAN && conda run -n jojo python align-faces.py --inputdir="'+self.temp_img_folder+'" --outputdir="'+self.img_folder+'"'
		#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
		os.system(cmd_align_faces)

	#-------------------------------------------------------------------------------------------
	def predict(self):
		# Prediction will be saved in result_file
		result_file = self.exp_folder+"/prediction_"+self.colormode+".txt"
		#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
		os.system('conda run -n model python predict.py --imagespath="'+self.img_folder+'" --resultfile="'+result_file+'" --modelpath="'+self.model_path+'" --colormode="'+self.colormode+"'")

		# Read and return the prediction contained in result_file
		f = open(result_file, "r")
		content = f.read()
		self.age = content.split(",")[0]
		self.gender = content.split(",")[1]
		f.close()
		return self.age, self.gender
	#-------------------------------------------------------------------------------------------
	def get_age(self):
		return self.age
	def get_gender(self):
		return self.gender
	#-------------------------------------------------------------------------------------------
	def predict_all_models(self):

		models_rgb = []
		models_rgb.append(os.path.join(self.root, "models/19fold/model_4"))
		models_rgb.append(os.path.join(self.root, "models/20netto/model_4"))
		models_rgb.append(os.path.join(self.root, "models/20pepper05/model_4"))

		models_gray = []
		models_gray.append(os.path.join(self.root, "models/19foldgray/model_4"))
		models_gray.append(os.path.join(self.root, "models/20nettogray/model_4"))
		models_gray.append(os.path.join(self.root, "models/20pepper05gray/model_4"))
		
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
	def behave(self):
		
		age_group = AgeGroups().getGroupFromAge(self.get_age())
		
		if age_group == 0:
			print()
		elif age_group == 1:
			print()
		elif age_group == 1:
			print()
		
		return None

#-------------------------------------------------------------------------------------------
if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--ip", type=str, default="172.20.10.10", help="Robot ip address.")
	parser.add_argument("--port", type=int, default=9559, help="Robot port number.")
	parser.add_argument("--colormode", help="rgb or grayscale", type=str, default="rgb")

	args = parser.parse_args()
	ip_robot = args.ip
	port_robot = args.port

	start_time = time.time()

	id_exp = 6
	#while True:
	exp = Experiment(id_exp, colormode=args.colormode)
	#exp.capture_images(ip=ip_robot, port=port_robot)
	#exp.align_images()
	exp.align_images_jojogan()
	# PREDICTION
	#age, gender = exp.predict()
	#print("Final age: ", age)
	#print("Final gender: ", gender)
	#	id_exp += 1

	print("--- %s seconds ---" % round(time.time() - start_time, 2))
	#exp.predict_all_models()


