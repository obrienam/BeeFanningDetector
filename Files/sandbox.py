import cv2
import numpy as np
from scipy import ndimage
times=0
vs=cv2.VideoCapture("/Users/aidanobrien/Documents/GitHub.nosync/BeeFanningDetector/Assets/test_img&videos/90test1.h264")
bk=cv2.imread('/Users/aidanobrien/Documents/GitHub.nosync/BeeFanningDetector/Assets/test_img&videos/90test1.png')
bk2=cv2.imread('/Users/aidanobrien/Documents/GitHub.nosync/BeeFanningDetector/Assets/test_img&videos/black.png')
#bk=bk[100:100+240,0:0+640]
#bk2=bk2[100:100+240,0:0+640]
while True:
    hasframes,img=vs.read()
    if(hasframes == False):
        break

    
    #img=img[100:100+240,0:0+640]
    cv2.imshow("frame",img)
    subImage=(bk.astype('int32')-img.astype('int32')).clip(0).astype('uint8')
    grey=cv2.cvtColor(subImage,cv2.COLOR_BGR2GRAY)
    retval,thresh=cv2.threshold(grey,35,255,cv2.THRESH_BINARY)
    kernel=np.ones((5,5),np.uint8)
    thresh=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)
    noback = cv2.bitwise_and(img, img, mask= thresh)
    #Color Bounds for 60 fps vids
    #upper = np.array([220,220,220])  
    #lower = np.array([160,160,160])  

    #Color Bounds for 30 fps vids
    upper = np.array([255,255,255])  
    lower = np.array([128,128,128])  
    mask = cv2.inRange(noback, lower, upper)

    wings = cv2.bitwise_and(noback, noback, mask=mask)
    cv2.imshow('No back',noback)
    cv2.imshow('Just_Wings/Shadows',wings)
    
    if times > 0:
        key=cv2.waitKey(0) & 0xFF
        #if q is pressed, stop loop
        if key == ord("c"):
            continue
        if key == ord("q"):
            break
    times = times + 1
vs.release()
cv2.destroyAllWindows()