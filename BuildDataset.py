import cv2
import os
import csv
#This program extracts features from the training data
#create a cvs file to store 
dataset = open('dataset.csv', 'wt')

#The directory has the training image set
#iterate for every file in the directory
for filename in os.listdir('img/night/'):
    img = cv2.imread(os.path.join('img/night/',filename))
    if img is not None:
        #if the file is an image
        #convert it to HSV
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #split the channels
        h, s, v = cv2.split(hsv_image)
        #calculate the value histogram
        hist_v = cv2.calcHist([v],[0],None,[25],[0,255])
        #calculate the hue histogram
        hist_h = cv2.calcHist([h],[0],None,[25],[0,65])
        #write to the CSV file
        writer = csv.writer(dataset)
        hv = int(sum(hist_v[15:24]))
        hh = int(sum(hist_h))
        writer.writerow([hv,hh])

dataset.close()
