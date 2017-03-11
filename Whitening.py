import cv2


#read the image file as grayscale
img = cv2.imread('low_contrast.jpg',cv2.IMREAD_GRAYSCALE)

#compute the mean and standard deviation of the image using meanStdDev function
(mean, std) = cv2.meanStdDev(img)
cv2.imshow('original',img)

#set the value of output as (ip-mean)/std
img = img-mean
img = img/std
cv2.imshow('after whitening',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
