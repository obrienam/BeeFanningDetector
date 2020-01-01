import cv2
import numpy as np
from matplotlib import pyplot as plt 

for i in range(0,3):
    img = cv2.imread('image'+str(i+1)+'.jpg',0)
    e = cv2.Canny(img,50,100)
    plt.subplot(3,2,(1+2*i)),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image ' + str(i+1)), plt.xticks([]), plt.yticks([])
    plt.subplot(3,2,(2+2*i)),plt.imshow(e, cmap = 'gray')
    plt.title('Edge Image ' + str(i+1)), plt.xticks([]), plt.yticks([])
plt.show()