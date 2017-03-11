import cv2
import random

#read the image as grayscale
CopyImage = cv2.imread('img/884291399_260.jpg',cv2.IMREAD_GRAYSCALE)

#initialize the no. of pixels of noise
nof_Salt = 2000
nof_Pepper = 2000
cv2.imshow('original',CopyImage)
#get the width and height of the image
Width = CopyImage.shape[0]
Height = CopyImage.shape[1]

#iterate nof_Salt times
for Salt in range(0,nof_Salt):
    #generate random co-ordinates
    RWidth = random.randrange(0, Width)
    RHeight = random.randrange(0, Height)
    #make the pixel white
    CopyImage[RWidth, RHeight] = 255

#iterate nof_Pepper times
for Pepper in range(0,nof_Pepper):
    #generate random co-ordinates
    RWidth = random.randrange(0, Width)
    RHeight = random.randrange(0, Height)
    #make the pixel black
    CopyImage[RWidth, RHeight] = 0

#apply median blur to the image
#5 is the size of kernel and should be greater than 1 and odd
blur = cv2.medianBlur(CopyImage,5)

cv2.imshow('noise',CopyImage)
cv2.imshow('filtered',blur)

cv2.waitKey(0)
cv2.destroyAllWindows()
