import cv2
import numpy as np
import random
from PIL import Image
w, h = 200, 200
data = np.zeros((h, w, 3), dtype=np.uint8)
num_lines =15


i = 0
width = random.randint(1,5)
#randomly generating lines
while i<num_lines:
    vert = random.randint(0,199)
    hori = random.randint(0,199)
    data[0:199, vert:vert+width] = 255
    data[hori:hori+width, 0:199] = 255
    i = i+1

img = Image.fromarray(data, 'RGB')
img.save('my.jpg')
img.show()

#define prewitt x operator
prewittX = np.array((
	[-1, 0, 1],
	[-1, 0, 1],
	[-1, 0, 1]), dtype="int")

#define prewitt y operator
prewittY = np.array((
	[-1, -1, -1],
	[0, 0, 0],
	[1, 1, 1]), dtype="int")
#read image as to grayscale
gray=cv2.imread("my.jpg",0)

#apply the prewitt filters
preX = cv2.filter2D(gray, -1, prewittX)
preY = cv2.filter2D(gray, -1, prewittY)
pre=preX+preY
cv2.imshow("prewittX",preX)
cv2.imshow("prewittY",preY)
cv2.imshow("prewitt",pre)


sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)

cv2.imshow('sobelx',sobelx)
cv2.imshow('original',gray)
cv2.imshow('sobely',sobely)

cv2.waitKey(0)
cv2.destroyAllWindows()