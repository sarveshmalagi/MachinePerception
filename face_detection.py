# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 14:59:14 2017

@author: sarvesh
"""

import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('C:/Users/sarvesh/Documents/haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('C:/Users/sarvesh/Documents/haarcascade_eye.xml')
count = 0
for filename in os.listdir('C:/Users/sarvesh/Downloads/portrait/original/'):
    img = cv2.imread(os.path.join('C:/Users/sarvesh/Downloads/portrait/original/',filename),cv2.IMREAD_COLOR)
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) >= 1:
            count=count+1
print count
#for (x,y,w,h) in faces:
#    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#    roi_gray = gray[y:y+h, x:x+w]
#    roi_color = img[y:y+h, x:x+w]
#    
#    eyes = eye_cascade.detectMultiScale(roi_gray)
#    for (ex,ey,ew,eh) in eyes:
#        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#
#cv2.imshow('img',img)
#k = cv2.waitKey(0)
#cv2.destroyAllWindows()