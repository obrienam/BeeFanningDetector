import cv2 as cv
import numpy as np
import sys

image=cv.imread(sys.argv[1])
n_image = np.zeros(image.shape, image.dtype)

alpha=1.0
beta=0

alpha=float(sys.argv[2])
beta=float(sys.argv[3])

for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        for c in range(image.shape[2]):
            n_image[y,x,c]=np.clip(alpha*image[y,x,c] + beta, 0, 255)

nm=str.split(sys.argv[1],'.jpg')
cv.imwrite(nm[0]+"contrast.jpg",n_image)