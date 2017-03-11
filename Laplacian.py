import cv2
import numpy as np

#read the input images
img=cv2.imread("img/lenna.png",0)
#img1=cv2.imread("img/star.png",0)

#Apply laplacian
res = cv2.Laplacian(img,cv2.CV_8U)
#res1 = cv2.Laplacian(img1,cv2.CV_8U)

#The sharpened image is the difference between img and laplacian
sharplap=img-res
cv2.imshow("originalImage",img)
cv2.imshow("laplacian",res)
cv2.imshow("sharplap",sharplap)
#sharplap=img-res
#sharplap1=img1+res1
#cv2.imshow("originalImage1",img1)
#cv2.imshow("laplacian1",res1)
#cv2.imshow("sharplap1",sharplap1)
cv2.waitKey(0)
cv2.destroyAllWindows()
