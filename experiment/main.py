import os
import sys
import argparse

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
		#cmd_capture_images = "python pepper_greeter.py --ip='"+str(ip)+"' --port="+str(port)+" --outputdir='"+self.temp_img_folder+"'"
		# Or it is blocked since beginning and capture images
		cmd_capture_images = "python pepper_capture_images.py --ip='"+str(ip)+"' --port="+str(port)+" --outputdir='"+self.temp_img_folder+"'"
		os.system(cmd_capture_images)

		# Crop face from the image and save it in outputdir
		cmd_align_faces = "cd JoJoGAN && conda run -n jojo python align-faces.py --inputdir='"+self.temp_img_folder+"' --outputdir='"+self.img_folder+"' --device='cpu'"
		#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
		os.system(cmd_align_faces)

	#-------------------------------------------------------------------------------------------
	def predict(self):
		# Prediction will be saved in result_file
		result_file = self.exp_folder+"/prediction_"+self.colormode+".txt"
		#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
		#print("python predict.py --imagespath='"+self.img_folder+"' --resultfile='"+result_file+"' --modelpath='"+self.model_path+"'")
		os.system("conda run -n model python predict.py --imagespath='"+self.img_folder+"' --resultfile='"+result_file+"' --modelpath='"+self.model_path+"' --colormode='rgb'")

		# Read and return the prediction contained in result_file
		f = open(result_file, "r")
		age = f.read().split[","][0]
		gender = f.read().split[","][1]
		f.close()
		return age, gender

#-------------------------------------------------------------------------------------------
if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--ip", type=str, default="172.20.10.10", help="Robot ip address.")
	parser.add_argument("--port", type=int, default=9559, help="Robot port number.")
	parser.add_argument("--colormode", help="rgb or grayscale", type=str, required=True)

	args = parser.parse_args()
	ip_robot = args.ip
	port_robot = args.port

	id_exp = 6
	#while True:
	exp = Experiment(id_exp, colormode=args.colormode)
	#exp.capture_images(ip=ip_robot, port=port_robot)
	age, gender = exp.predict()
	print("Final age: ", age)
	print("Final gender: ", gender)
	#	id_exp += 1

