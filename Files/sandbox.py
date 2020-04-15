import cv2
import numpy as np
from scipy import ndimage
times=0
vs=cv2.VideoCapture("/Users/aidanobrien/Documents/GitHub/BeeFanningDetector/Assets/test_img&videos/test_vid5.mp4")
bk=cv2.imread("/Users/aidanobrien/Documents/GitHub/BeeFanningDetector/Assets/test_img&videos/test_bk2.png")
while True:
    hasframes,image=vs.read()
    #image=image[175:175+230,0:0+640]
    #bk=bk[175:175+230,0:0+640]
    subImage=(bk.astype('int32')-image.astype('int32')).clip(0).astype('uint8')
    grey=cv2.cvtColor(subImage,cv2.COLOR_BGR2GRAY)
    retval,thresh=cv2.threshold(grey,35,255,cv2.THRESH_BINARY_INV)
    thresh=255-thresh
    kernel=np.ones((5,5),np.uint8)
    thresh=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)
    thresh=thresh[175:175+230,0:0+640]
    cv2.imshow("Output", thresh)
    if times > 0:
        key=cv2.waitKey(1) & 0xFF
        #if q is pressed, stop loop
        if key == ord("c"):
            continue
        if key == ord("q"):
            break
    times = times + 1
    