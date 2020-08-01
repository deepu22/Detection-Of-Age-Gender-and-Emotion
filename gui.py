import shutil,os
from os import listdir
from PIL import ImageTk,Image
from pathlib import Path
from tkinter import *
import tkinter.messagebox
from tkinter import filedialog

# -------------------------------------------------
# from __future__ import division
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.models import model_from_json
import numpy
# import os
import numpy as np
import cv2
# -------------------------------------------------

# define-----------------------
json_file = open('model/m.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
 #load weights into new model
loaded_model.load_weights("model/m.h5")

#loaded_model=load_model('D:/FInal year project/my project/model/emotion.h5')

print("Loaded model from disk")

#setting image resizing parameters
WIDTH = 48
HEIGHT = 48
x=None
y=None
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# -----------------



def add_image():
	#os.remove(r'D:/FInal year project/my project/d.jfif')
	# os.system('python upload_image.py')
	print("add image")

	root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
	print (root.filename)

	full_size_image = cv2.imread(root.filename)
	print("Image Loaded")
	gray=cv2.cvtColor(full_size_image,cv2.COLOR_RGB2GRAY)
	face = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
	#faces = face.detectMultiScale(gray, 1.3  , 5)
	#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
	#faces = face.detectMultiScale(gray,1.3 ,minNeighbors=5,minSize=(1,1),flags = cv2.CASCADE_SCALE_IMAGE)
	faces = face.detectMultiScale(
			gray,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=(30, 30)
	)
	#detecting faces
	for (x, y, w, h) in faces:
	        roi_gray = gray[y:y + h, x:x + w]
	        #roi_color = img[y:y+h, x:x+w]
	        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
	        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
	        cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (255,0, 0), 2)
	        #cv2.rectangle(full_size_image, (10,10), (220,220), (0, 255, 0), 1)
	        #predicting the emotion
	        yhat= loaded_model.predict(cropped_img)
	        cv2.putText(full_size_image, labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255, 0), 1, cv2.LINE_AA)
	        print("Emotion: "+labels[int(np.argmax(yhat))])
	dim=(750,750)
	resized = cv2.resize(full_size_image, dim, interpolation = cv2.INTER_AREA)
	 
	cv2.imshow('Emotion', resized)
	cv2.waitKey()

def recreate1():
	os.system('cmd /c "python agegender.py"')

def recreate():
	os.system('cmd /c "python videotest.py"')

def destroy():
	root.destroy()

root = Tk()

root.geometry('800x500')
root.title("Final Year Project")

msg = Message(root, width='800',font=('times', 20, 'bold'), bd='5',text = 'Detection of Age Gender and Emotion')
msg.place(x=150, y=50)

b1 = Button(root ,text='Upload the Image',bg='#C0C0C0',height=5, width=15,command=add_image)
b1.place(x=100, y=200)


b3 = Button(root ,text='Detect Emotion',bg='#C0C0C0', height=5, width=15,command=recreate)
b3.place(x=250, y=200)

b5=Button(root,text='Detect Age & Gender',bg='#C0C0C0',height=5,width=15,command=recreate1)
b5.place(x=400, y=200)
b4 = Button(root ,text='Exit Gui', height=5,bg='#C0C0C0', width=15,command=destroy)
b4.place(x=550, y=200)

root.mainloop()