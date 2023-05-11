'''
AgeGroupBehave acts a behavior on Pepper robot based on given person's age and gender.
It is not used its exaclty age but it is derived an age-group which definition is in "age_groups.py" file. 
In all cases, the behavior is an oral explanation and an image visualization on the tablet. 
Pepper's tags used for animated speech are taken from:
http://doc.aldebaran.com/2-5/naoqi/motion/alanimationplayer-advanced.html#animationplayer-list-behaviors-pepper
'''
#!/usr/bin/python
# -*- coding: UTF-8 -*-

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
		self.age_group = int(AgeGroups().getGroupFromAge(age))
		if self.age_group == 2 and age >= 18:
			self.age_group == 3
		self.gender = gender

		app.start()
		session = app.session
		# Get some services
		self.memory = session.service("ALMemory")
		self.leds_service = session.service("ALLeds")
		self.tts = session.service("ALTextToSpeech")
		self.animated_speech = session.service("ALAnimatedSpeech")
		self.tablet_service = session.service("ALTabletService")

		self.tts.setLanguage("Italian")
		self.speech_speed = "70"

		application_id = "thesis-giorgia-e2b93a"
		self.root_image_path = "http://198.18.0.1/apps/"+application_id+"/"

		self.tablet_service.hideImage()
	
	#-------------------------------------------------------------------------------------------
	def say_something(self, animation_tag, text):
		"""
		Say the given 'text' with the animation from 'animation_tag'. 
		"""
		sentence = "\RSPD="+self.speech_speed+"\ "
		sentence += "^start(%s) %s ^wait(%s)" % (animation_tag, text, animation_tag)
		sentence +=  "\RST\ "
		self.animated_speech.say(sentence)

    #-------------------------------------------------------------------------------------------
	def test_behave(self):
		
		# Farlo stare fermo da quando intercetta una persona??
		print("Test animations.")
		
		# Show image on Tablet
		image_path = self.root_image_path+"3_0_female.jpg"
		self.tablet_service.preLoadImage(image_path)
		self.tablet_service.showImage(image_path)

		# Hide the web view
		self.tablet_service.hideImage()

		# Led and animation during dialog
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
		text = "Hello! I change the color of my eyes."
		# Set the eye color and start the animation
		#self.leds_service.fadeRGB("FaceLeds", 1, 0, 0, duration)
		self.animated_speech.say("^start(%s) %s ^wait(%s)" % (animation_name, text, animation_name))
		# Reset the eye color after the animation has finished
		time.sleep(2)  # Wait for the animation to finish
		self.leds_service.randomEyes(5)
		self.leds_service.fadeRGB("FaceLeds", 1, 1, 1, duration)
	
	#-------------------------------------------------------------------------------------------
	def behave(self):
		"""
		Reproduce a behavior on Pepper according to the person's age group..
		"""
		# 0-2 or 3-17
		if self.age_group == 0 or self.age_group == 1 or self.age_group == 2 : 
			print("Under 18: bad.")
		# 18-29 because I know that people coming in the lab are over 18
		elif self.age_group == 3: 
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

		# Final greetings
		text = "Grazie per aver partecipato a questo esperimento. \n"
		text += "Spero che l'esperienza sia stata interessante e utile per te. \n" 
		text += "I dati raccolti ci aiuteranno a migliorare le nostre tecnologie e a creare soluzioni sempre pi"+u'\xf9'+" avanzate per il futuro. \n"
		text += "Ti ringrazio ancora per la tua partecipazione e ti auguro una buona giornata."
		self.say_something(animation_tag = "enthusiastic", text=text)

	#-------------------------------------------------------------------------------------------
	def case_3(self): # 18-29 years
		"""
		Behavior for persons aged between 18 and 29.
		"""
		rand = random.randint(0, 1)
		image_path = self.root_image_path+"/3_"+str(rand)+"_"+str(self.gender)+".jpg"
		# OBAMA
		if rand == 0:
			text = "Oggi voglio ricordare con te un evento storico del 2008: il primo presidente di colore. \n"
			text += "Il primo presidente di colore della storia "+u'\xe8'+" stato Barack Obama, che "+u'\xe8'+" stato eletto presidente degli Stati Uniti nel novembre del 2008 e ha servito due mandati consecutivi fino al gennaio del 2017.\n"
			text += "Obama "+u'\xe8'+" stato il quarantaquattresimo presidente degli Stati Uniti e il primo afroamericano ad occupare la Casa Bianca.\n"
			text += "L'elezione di Obama ha rappresentato un momento significativo nella storia degli Stati Uniti e ha segnato un importante passo avanti nella lotta per l'uguaglianza razziale. La sua elezione "+u'\xe8'+" stata salutata da molti come una vittoria per la giustizia sociale e ha ispirato molti altri paesi in tutto il mondo."
		# GIOCO: winx or super mario
		else:
			text = "Oggi voglio parlarti di cos'"+u'\xe8'+" l'infanzia e cosa potresti aver incontrato nella tua.\n"
			text += "L'infanzia pu"+u'\xf2'+" essere definita come la fase dello sviluppo umano che va dalla nascita all'adolescenza, solitamente compresa tra i 0 e i 12-13 anni di et"+u'\xe0'+". A livello psicologico, l'infanzia "+u'\xe8'+" una fase critica dello sviluppo, durante la quale si sviluppano alcune delle abilit"+u'\xe0'+" sociali, emotive e cognitive fondamentali che influenzeranno la vita adulta.\n"
			text += "Della tua infanzia ricorderai Super Mario, i Pokemon, le Winx e Hannah Montana."
		# Show image
		self.tablet_service.preLoadImage(image_path)
		self.tablet_service.showImage(image_path)
		# Start speaking
		self.say_something(animation_tag = "explain", text=text)
		# Hide image
		self.tablet_service.hideImage()

	def case_4(self): # 30-39 years
		"""
		Behavior for persons aged between 30 and 39.
		"""
		rand = random.randint(0, 1)
		# TORRI GEMELLE
		if rand == 0:
			image_path = self.root_image_path+"/4_0.jpg"
			text = "Eri "
			text += "piccolo" if self.gender=="male" else "piccola"
			text += "ma sicuramente ti ricorderai cos'"+u'\xe8'+" successo l'11 settembre 2001.\n"
			text += "L'11 settembre 2001 c'"+u'\xe8'+" stato un attacco terroristico che ha sconvolto il mondo intero. In quell'occasione, 19 terroristi di Al Qaida fecero schiantare due aerei Boeing 767 contro le Torri Gemelle del World Trade Center di New York\n"
			text += "L'attacco caus"+u'\xf2'+" la morte di quasi tremila persone e fer"+u'\xec'+" oltre sei mila. L'attentato ha avuto un impatto enorme sulla politica internazionale e sulla sicurezza aerea, portando a una serie di riforme e cambiamenti."
		# LADY D
		else:
			image_path = self.root_image_path+"/4_1_"+str(self.gender)+".jpg"
			text = "Oggi voglio ricordare con te un personaggio storico vissuto fino al 1997\n"
			text += "Lady D "+u'\xe8'+" il soprannome con cui era comunemente conosciuta Lady Diana Spencer, nata il primo luglio 1961 nel Regno Unito. "
			text += ""+u'\xe8'+" stata una figura pubblica molto amata. Lady Diana "+u'\xe8'+" stata sposata con il principe Carlo, erede al trono britannico, dal 1981 al 1996. "
			text += "Dopo il divorzio dal principe Carlo, Lady Diana ha continuato il suo lavoro di beneficenza, diventando un'importante sostenitrice della lotta contro l'AIDS, delle cause umanitarie e della protezione delle mine antiuomo. "
			text += "Nel 1997, Lady Diana "+u'\xe8'+" deceduta tragicamente in un incidente automobilistico a Parigi, all'et"+u'\xe0'+" di soli 36 anni. La sua morte ha suscitato un'enorme reazione emotiva in tutto il mondo, con molte persone che hanno espresso il loro cordoglio per la sua scomparsa."
		# Show image
		self.tablet_service.preLoadImage(image_path)
		self.tablet_service.showImage(image_path)
		# Start speaking
		self.say_something(animation_tag = "explain", text=text)
		# Hide image
		self.tablet_service.hideImage()

	def case_5(self): # 40-49 years
		"""
		Behavior for persons aged between 40 and 49.
		"""
		rand = random.randint(0, 1)
		image_path = self.root_image_path+"/5_"+str(rand)+"_"+str(self.gender)+".jpg"
		# MICHAEL JACKSON
		if rand == 0:
			text = "Oggi voglio rivelarti alcune curiosit"+u'\xe0'+" su Michael Jackson\n"
			text += "Jackson aveva una passione per i parchi divertimenti e voleva costruire il suo proprio parco divertimenti chiamato Neverland. La propriet"+u'\xe0'+" era ispirata all'isola immaginaria di Peter Pan, e comprendeva attrazioni come una ruota panoramica, una montagna russa e un treno fantasma\n"
			text += "Inoltre, durante la registrazione del brano Beat It, Jackson ha fatto una scommessa con Eddie Van Halen, il chitarrista che ha suonato la chitarra solista nella canzone. Jackson gli ha chiesto di suonare il riff di chitarra pi"+u'\xf9'+" veloce possibile, e gli ha promesso che se ci fosse riuscito, gli avrebbe dato una Porsche. Van Halen ha fatto il riff perfettamente e Jackson ha mantenuto la sua promessa regalando a Van Halen una Porsche."
		# MURO DI BERLINO
		else:
			text = "Oggi voglio ricordare con te un evento storico del 1989: la caduta del Muro di Berlino\n"
			text += "La caduta del Muro di Berlino nel 1989 "+u'\xe8'+" stato uno degli eventi pi"+u'\xf9'+" significativi del ventesimo secolo, in quanto ha segnato la fine della Guerra Fredda e l'inizio di un nuovo capitolo nella storia mondiale. Il Muro di Berlino, costruito nel 1961, era diventato il simbolo pi"+u'\xf9'+" potente della divisione del mondo in due blocchi contrapposti, rappresentando la separazione tra la Germania Est e la Germania Ovest e l'ideologia comunista dell'Unione Sovietica\n"
			text += "Il 9 novembre 1989, il governo della Germania Est annunci"+u'\xf2'+" che i suoi cittadini sarebbero stati autorizzati a viaggiare liberamente e quella stessa notte, le persone iniziarono a demolire il Muro."
		# Show image
		self.tablet_service.preLoadImage(image_path)
		self.tablet_service.showImage(image_path)
		# Start speaking
		self.say_something(animation_tag = "explain", text=text)
		# Hide image
		self.tablet_service.hideImage()

	def case_6(self): # 50-59 years
		"""
		Behavior for persons aged between 50 and 59.
		"""
		rand = random.randint(0, 1)
		image_path = self.root_image_path+"/6_"+str(rand)+"_"+str(self.gender)+".jpg"
		# RITORNO AL FUTURO
		if rand == 0:
			text = "Oggi voglio ricordare con te un film del 1985. Non ti dir"+u'\xf2'+" quale ma lo capirai\n"
			text += "Questo film "+u'\xe8'+" stato un grande successo anche in Italia, tanto che ha ispirato una serie televisiva italiana intitolata Torno indietro e cambio vita, trasmessa su Rai 1 nel 2015. "
			text += "Ha avuto bisogno di cinque anni per essere scritto e prodotto ma alla fine, ha vinto l'Oscar per il miglior montaggio sonoro. "
			text += "Un elemento chiave del film "+u'\xe8'+"la DeLorean che "+u'\xe8'+"stata scelta perch"+u'\xe9'+" la sua forma ricorda quella di un'astronave\n"
			text += "Forse l'avrei capito ma si, sto proprio parlando di Ritorno al futuro."
		# PRIMA TV A COLORI
		else:
			text = "Oggi voglio ricordare con te un evento storico che ha cambiato per sempre il modo di vedere della gente: la televisione a colori\n"
			text += "In Italia, l'avvento della prima televisione a colori "+u'\xe8'+" stato percepito come un evento molto atteso e importante. La prima trasmissione televisiva a colori in Italia avvenne nel 1977, quando RAI 1 trasmise la finale del Festival di Sanremo." 
			text += "Il pubblico italiano rimase affascinato dalle immagini a colori, che davano una nuova vita ai programmi televisivi e facevano sembrare la trasmissione ancora pi"+u'\xf9'+" vicina e realistica. Tuttavia, la televisione a colori non era subito accessibile a tutti, poich"+u'\xe9'+" le televisioni a colori erano ancora molto costose e non erano disponibili per la maggior parte delle persone\n"
			text += "Fu solo negli anni 80 che la televisione a colori divenne pi"+u'\xf9'+" diffusa e accessibile al grande pubblico in Italia, grazie anche alla produzione di televisori a prezzi pi"+u'\xf9'+" accessibili."

		# Show image
		self.tablet_service.preLoadImage(image_path)
		self.tablet_service.showImage(image_path)
		# Start speaking
		self.say_something(animation_tag = "explain", text=text)
		# Hide image
		self.tablet_service.hideImage()

	def case_7(self): # 60-69 years
		"""
		Behavior for persons aged between 60 and 69.
		"""
		rand = random.randint(0, 2)
		image_path = self.root_image_path+"/7_"+str(rand)+"_"+str(self.gender)+".jpg"
		# PATTY PRAVO
		if rand == 0:
			text = "Ci sono molti cantanti famosi italiani della tua stessa generazione\n"
			text += "Una fra questi, "+u'\xe8'+" Patty Pravo nata il 9 aprile 1948. Il suo primo grande successo "+u'\xe8'+" arrivato nel 1966 con la canzone Ragazzo triste, che ha raggiunto la vetta delle classifiche italiane. Da allora, ha continuato a sfornare una serie di grandi successi, tra cui: La bambola, Pazza idea, E dimmi che non vuoi morire, Pensiero stupendo e molti altri\n"
			text += "Patty Pravo ha fatto la storia della musica italiana grazie alla sua voce unica e al suo stile innovativo, che ha unito elementi del rock, della pop e della musica classica. Ha anche sperimentato con i costumi e il trucco, diventando un'icona della moda dell'epoca."
		# GIANNI MORANDI
		elif rand == 1:
			text = "Ci sono molti cantanti famosi italiani della tua stessa generazione\n"
			text += "Uno fra questi "+u'\xe8'+" Gianni Morandi nato il 11 dicembre 1944. Morandi ha iniziato la sua carriera musicale nel 1962, a soli 17 anni, partecipando al Festival di Sanremo con la canzone Fatti mandare dalla mamma, che ha ottenuto un grande successo di pubblico. Da quel momento, ha continuato a pubblicare una serie di grandi successi, tra cui In ginocchio da te, Scende la pioggia, Non son degno di te e molti altri\n"
			text += "Oltre alla musica, Morandi ha anche avuto una carriera come attore, recitando in diversi film, spettacoli teatrali e programmi televisivi. "+u'\xe8'+" stato anche il conduttore di Canzonissima, uno dei pi"+u'\xf9'+" famosi programmi televisivi italiani degli anni 70."
		# PRIMO CELLULARE
		else:
			text = "Oggi voglio ricordare con te un evento storico che ha cambiato per sempre il modo di comunicare della gente: la prima chiamata con cellulare\n"
			text += "La prima chiamata con un cellulare "+u'\xe8'+" stata effettuata il 3 aprile 1973 dall'ingegnere Martin Cooper della Motorola, che all'epoca era il responsabile dello sviluppo del primo telefono cellulare al mondo, il Motorola DynaTAC. La chiamata "+u'\xe8'+" stata effettuata sulla Quinta Strada a New York, davanti all'hotel Hilton, e ha coinvolto il rivale della Motorola che lavorava per la Bell Labs\n"
			text += "La chiamata "+u'\xe8'+" stata effettuata con il primo prototipo del telefono cellulare, che pesava circa 1 kilo e aveva una durata della batteria di soli 20 minuti. Il telefono aveva una forma simile a quella di un grosso mattone e poteva essere utilizzato solo per effettuare chiamate telefoniche."
		# Show image
		self.tablet_service.preLoadImage(image_path)
		self.tablet_service.showImage(image_path)
		# Start speaking
		self.say_something(animation_tag = "explain", text=text)
		# Hide image
		self.tablet_service.hideImage()

	def case_8(self): # over 70
		"""
		Behavior for persons aged over 70.
		"""
		image_path = self.root_image_path+"/8_0_"+str(self.gender)+".jpg"
		# SBARCO SULLA LUNA
		text = "Oggi voglio ricordare con te un evento storico del 1969: lo sbarco sulla Luna.\n"
		text += "Lo sbarco sulla Luna del 20 luglio 1969 "+u'\xe8'+" stato un evento storico che ha catturato l'immaginazione del mondo intero.\n"
		text += "Ecco alcune curiosit"+u'\xe0'+" sullo sbarco sulla Luna.\n"
		text += "Il modulo lunare non aveva un sistema di rampa, quindi gli astronauti dovevano usare una scala per scendere dal modulo sulla superficie lunare. La scala fu progettata da un ingegnere italiano di nome Giovanni Battista Piumatti.\n"
		text += "Inoltre, nel 2021, l'Agenzia Spaziale Europea ha pubblicato una mappa della superficie lunare in 3D ad alta risoluzione, che includeva i luoghi dove si "+u'\xe8'+" svolta la missione Apollo 11. La mappa consente ai visitatori di esplorare i luoghi dove gli astronauti hanno camminato sulla superficie della Luna, offrendo un'esperienza quasi realistica."
		# Show image
		self.tablet_service.preLoadImage(image_path)
		self.tablet_service.showImage(image_path)
		# Start speaking
		self.say_something(animation_tag = "explain", text=text)
		# Hide image
		self.tablet_service.hideImage()

#-------------------------------------------------------------------------------------------
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--ip", type=str, default="172.20.10.10", help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
	parser.add_argument("--port", type=int, default=9559, help="Naoqi port number")
	parser.add_argument("--age", help="estimated person age", type=int, required=True)
	parser.add_argument("--gender", help="estimated person gender ('female' or 'male')", required=True)

	args = parser.parse_args()
	try:
		# Initialize qi framework.
		connection_url = "tcp://" + args.ip + ":" + str(args.port)
		app = qi.Application(["AgeGroupBehave", "--qi-url=" + connection_url])
	except RuntimeError:
		print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
		"Please check your script arguments. Run with -h option for help.")
		sys.exit(1)

	human_greeter = AgeGroupBehave(app=app, age=args.age, gender=args.gender)

	human_greeter.behave()
