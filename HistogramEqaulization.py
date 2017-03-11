import cv2
#read the input image
img = cv2.imread('img/low_contrast1.jpg')
#convert to grayscale
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#perform histogram equalization
cv2.equalizeHist(gray)

#display image
cv2.imshow("equalizeHist", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
