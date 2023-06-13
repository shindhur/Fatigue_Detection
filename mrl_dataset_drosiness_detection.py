import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

lbl=['Close','Open']

model = load_model('models/cnncat2.h5')
path = os.getcwd()
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

imagePath = 'eyes.jpg'

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('img',image)
left_eye = leye.detectMultiScale(gray)
right_eye =  reye.detectMultiScale(gray)

for (x,y,w,h) in right_eye:
    r_eye=image[y:y+h,x:x+w]
    count=count+1
    r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
    r_eye = cv2.resize(r_eye,(24,24))
    r_eye= r_eye/255
    r_eye=  r_eye.reshape(24,24,-1)
    r_eye = np.expand_dims(r_eye,axis=0)
    rpred = (model.predict(r_eye)>0.5).astype('int32')
    print(rpred)
    if(rpred[0][1]==1).all():
        lbl='Open' 
    if(rpred[0][1]==0).all():
        lbl='Closed'
    break

if(((rpred[0][1]==0).all())):
    score=score+1
    print('closed')
    cv2.putText(image,'closed',(100,100-20), font, 1,(255,255,255),1,cv2.LINE_AA)
# if(rpred[0]==1 or lpred[0]==1):
else:
    score=score-1
    print('open')
    cv2.putText(image,'open',(100,100-20), font, 1,(255,255,255),1,cv2.LINE_AA)

cv2.imwrite(os.path.join(path,'image1.jpg'),image)
    
cv2.destroyAllWindows()
