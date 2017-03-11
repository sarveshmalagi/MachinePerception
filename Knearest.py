import cv2
import numpy as np
import matplotlib.pyplot as plt

#This function reads an image and returns the histogram values of hue and value
def getImageHist():
    ip_img = cv2.imread('img/test.jpg',cv2.IMREAD_COLOR)
    #resize the image to match the training set
    ip_img = cv2.resize(ip_img, (100,100))
    #convert image to HSV
    hsv_image = cv2.cvtColor(ip_img, cv2.COLOR_BGR2HSV)
    #split the HSV image
    h, s, v = cv2.split(hsv_image)
    #calculate the sum of values
    hist_v = cv2.calcHist([v],[0],None,[25],[0,255])
    #calculate the sum of hues in red and yellow range
    hist_h = cv2.calcHist([h],[0],None,[25],[0,65])
    
    return sum(hist_v[15:24]),sum(hist_h)

#This function uses CascadeClassifier to detect faces
def detectFace():
    #Read the Haar cascade file
    face_cascade = cv2.CascadeClassifier('img/haarcascade_frontalface_default.xml')
    #read the image and convert it into grayscale
    img = cv2.imread('img/test.jpg',cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #detect faces and store in array
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) >= 1:
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow('img',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return len(faces)
    

#if a face is detected, the image is a portrait, otherwise it maybe night or a landscape
if detectFace() >= 1:
    print 'Portrait'
else:
    #image is a landscape or night
    #load the training data in csv file to 2d array 'trainData'
    trainData = np.random.randint(0,100,(610,2)).astype(np.float32)
    raw_data = np.genfromtxt('dataset.csv', dtype=float, delimiter=',', names=True)
    r = 0
    for i in range(0,610):
        trainData[r,0] = raw_data[i][0]
        trainData[r,1] = raw_data[i][1]
        r=r+1

    #load the labels for training data
    responses = np.random.randint(0,2,(610,1)).astype(np.float32)
    responses[:305] = 0
    responses[306:] = 1
    
    #plot the landscapes in the training data in red color
    red = trainData[responses.ravel()==0]
    plt.scatter(red[:,0],red[:,1],80,'r','^')
    
    #plot the night images in training data in blue color
    blue = trainData[responses.ravel()==1]
    plt.scatter(blue[:,0],blue[:,1],80,'b','s')
    
    #plt.show()
    
    #get the features from the input image to be classified
    newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)
    newcomer[0,0],newcomer[0,1] = getImageHist()
    plt.scatter(newcomer[:,0],newcomer[:,1],80,'g','o')
    plt.show()
    #use the KNearest_create() function to create a K Neareast Neighbour model
    knn = cv2.ml.KNearest_create()
    #Train the model using training data and labels
    knn.train(trainData,cv2.ml.ROW_SAMPLE,responses)
    #get the nearest neighbour using the findNearest() method
    #The 2nd parameter is the 'K' in K nearest neighbour algorithm
    ret, results, neighbours ,dist = knn.findNearest(newcomer, 10)
    
    if results[0] == 0:
        print 'Landscape'
    else:
        print 'night'
