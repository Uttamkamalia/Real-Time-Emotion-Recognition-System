import cv2
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import math
import tensorflow as tf
from newcnn1 import *


def draw_curves(list1,frame):
    for i in range(0,len(list1)-1):
        cv2.line(frame,(list1[i][0],list1[i][1]),(list1[i+1][0],list1[i+1][1]),(0,0,255),2)


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))

preDICT = {0:"Angry",1:"Disgust",2:"Fear",3:"Happy",4:"Sad",5:"Surprize",6:"Neutral"}

webcam = cv2.VideoCapture(0)
#face_cas = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt_tree.xml")
face_cas = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor(r"C:\Users\acer\Desktop\emotion detection\shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


savingpath = "/tmp/emotion_nor/conweights"
session = tf.Session()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()

saver.restore(sess=session, save_path=savingpath)

while True:
        
        ret,frame = webcam.read()
        emotion = "unknown"
       
        #gray =  imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        faces = face_cas(gray, 1)
        
        x1,y,w,h=0,0,0,0
        for (i,face) in enumerate(faces):

            landmarks = predictor(gray,face)
            landmarks = face_utils.shape_to_np(landmarks)
            (x1,y,w,h) = face_utils.rect_to_bb(face)

            facecrop = gray[y:y+h,x1:x1+w]
            facecrop = clahe.apply(facecrop)
            facecrop = cv2.resize(facecrop,(48,48),interpolation = cv2.INTER_AREA)
            
            #facecrop = cv2.cvtColor(facecrop,cv2.COLOR_BGR2GRAY)
            facecrop = np.reshape(facecrop,(48,48,1))
            data = []
            data.append(facecrop)
            facecrop = np.array(data)
            print(facecrop)
           
            feed_dict = {x: facecrop, y_true: np.array([[1,0,0,0,0,0,0]])}

             #Calculate the predicted class using TensorFlow.
            cls_pred = session.run(y_pred_cls, feed_dict=feed_dict)

            emotion = preDICT[cls_pred.tolist()[0]]
            
            cv2.rectangle(frame,(x1,y),(x1+w,y+h),(0,255,0),3)
            cv2.putText(frame, emotion, (x1 - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)     
               
        #cv2.imshow("croped",facecrop)
        cv2.imshow("frame",frame)
        
        #cv2.waitKey()
        if cv2.waitKey(30) & 0xFF == 27:
             break


webcam.release()
cv2.destroyAllWindows()




            
            

    

