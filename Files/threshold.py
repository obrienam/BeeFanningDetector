import cv2
import numpy as np
bk = cv2.imread("/Users/aidanobrien/Documents/GitHub/BeeFanningDetector/Assets/test_bkgrnd.jpg")
img = cv2.imread("/Users/aidanobrien/Documents/GitHub/BeeFanningDetector/Assets/test_fan.jpg")
subImage=(bk.astype('int32')-img.astype('int32')).clip(0).astype('uint8')
grey=cv2.cvtColor(subImage,cv2.COLOR_BGR2GRAY)
retval,thresh=cv2.threshold(grey,35,255,cv2.THRESH_BINARY_INV)
thresh=255-thresh
kernel=np.ones((5,5),np.uint8)
thresh=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)
cv2.imwrite("../Assets/test_thresh.jpg",thresh)