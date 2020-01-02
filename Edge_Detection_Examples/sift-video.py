import cv2
import matplotlib.pyplot as plt
import math
import numpy as np

def countStill(img1, img2):
    m=0
    bk = cv2.imread('Assets/background-2.jpg')
    subImage1=(bk.astype('int32')-img1.astype('int32')).clip(0).astype('uint8')
    grey1=cv2.cvtColor(subImage1,cv2.COLOR_BGR2GRAY)
    retval1,thresh1=cv2.threshold(grey1,35,255,cv2.THRESH_BINARY_INV)
    img1=grey1
    cv2.imshow("thresh",thresh1)
    subImage2=(bk.astype('int32')-img2.astype('int32')).clip(0).astype('uint8')
    grey2=cv2.cvtColor(subImage2,cv2.COLOR_BGR2GRAY)
    retval2,thresh2=cv2.threshold(grey2,35,255,cv2.THRESH_BINARY_INV)
    img2=grey2

    #sift
    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)
    kp1=keypoints_1
    kp2=keypoints_2
    d_1=descriptors_1
    d_2=descriptors_2
    #feature matching
    bf = cv2.BFMatcher()

    matches = bf.match(d_1,d_2)
    matches = sorted(matches, key = lambda x:x.distance)

    for i in range(len(kp1)):
        for j in range(len(kp2)):
            #print(kp2[j].pt[0])
            #print(kp2[j].pt[1])
            
            if(abs(kp1[i].pt[0]-kp2[j].pt[0])<2 and abs(kp1[i].pt[1]-kp2[j].pt[1])<4):
                m+=1
                break
        
    pt=keypoints_1[0].pt
    return m

vs=cv2.VideoCapture("/Users/aidanobrien/Documents/GitHub/BeeFanningDetector/Assets/bees2.mp4")
#vs=cv2.VideoCapture("C:/Users/obrienam/Documents/GitHub/CV_Research/Assets/contrast.mp4")
firstFrame=None
prev_s=None
bev="Steady"
s=0
frame_width = int(vs.get(3))
frame_height = int(vs.get(4))
while True:
    hasFrames,frame=vs.read()
 #   hasFrames2,oframe=fs.read()
    if (hasFrames==False):
        break
    
    if firstFrame is not None:
        #frame=upContrast(frame,2,1)
        s=countStill(firstFrame,frame)
        print(s)
        if(prev_s is None):
            bev="Steady"
        elif(prev_s<s and abs(s-prev_s)>=15):  
            bev="Increasing"
        elif(prev_s==s or abs(s-prev_s)<15):
            bev="Steady" 
        elif(prev_s>s and abs(s-prev_s)>=15):
            bev="Decreasing"   
        prev_s=s
        firstFrame=frame
    else:    
        firstFrame=frame
    key=cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    
    
    cv2.putText(frame, "Stationary Bees: {}".format(bev), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("vid",frame)
vs.release()
cv2.destroyAllWindows()