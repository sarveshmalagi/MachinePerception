import cv2
import numpy as np

#read the image as grayscale
img = cv2.imread('img/noise.jpg',cv2.IMREAD_GRAYSCALE)

#Apply gaussian blur using the GaussianBlur function
#The 2nd parameter (15,15) is the kernel size
#The 3rd and 4th parameters are the standard deviations in x and y. Increasing deviations will increase the blur
blur = cv2.GaussianBlur(img,(15,15),5,3)

#increase the scale
blur_inc = cv2.GaussianBlur(img,(15,15),50,100)

cv2.imshow('img',img)
cv2.imshow('blur',blur)
cv2.imshow('blur_inc',blur_inc)

cv2.waitKey(0)
cv2.destroyAllWindows()
