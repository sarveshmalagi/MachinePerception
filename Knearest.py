# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 12:22:27 2017

@author: sarvesh
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def getImage():
    ip_img = cv2.imread('test.jpg',cv2.IMREAD_COLOR)
    ip_img = cv2.resize(ip_img, (100,100))
    hsv_image = cv2.cvtColor(ip_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    hist_v = cv2.calcHist([v],[0],None,[25],[0,255])
    hist_h = cv2.calcHist([h],[0],None,[25],[0,65])
    return sum(hist_v[15:24]),sum(hist_h)

def detectFace():
    face_cascade = cv2.CascadeClassifier('C:/Users/sarvesh/Documents/haarcascade_frontalface_default.xml')    
    img = cv2.imread('test.jpg',cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return len(faces)
    
    
if detectFace() >= 1:
    print 'Portrait'
else:

    trainData = np.random.randint(0,100,(610,2)).astype(np.float32)
    raw_data = np.genfromtxt('dataset.csv', dtype=float, delimiter=',', names=True)
    r = 0
    for i in range(0,610):
        trainData[r,0] = raw_data[i][0]
        trainData[r,1] = raw_data[i][1]
        r=r+1

    responses = np.random.randint(0,2,(610,1)).astype(np.float32)
    responses[:305] = 0
    responses[306:] = 1
    
    red = trainData[responses.ravel()==0]
    plt.scatter(red[:,0],red[:,1],80,'r','^')
    
    
    blue = trainData[responses.ravel()==1]
    plt.scatter(blue[:,0],blue[:,1],80,'b','s')
    
    plt.show()
    
    newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)
    newcomer[0,0],newcomer[0,1] = getImage()
    
    
    knn = cv2.ml.KNearest_create()
    knn.train(trainData,cv2.ml.ROW_SAMPLE,responses)
    ret, results, neighbours ,dist = knn.findNearest(newcomer, 10)
    
    if results[0] == 0:
        print 'Landscape'
    else:
        print 'night'
