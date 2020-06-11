import cv2
import numpy as np
from scipy import ndimage
times=0
numfan=0
vs=cv2.VideoCapture("/Users/aidanobrien/Documents/GitHub/BeeFanningDetector/Assets/test_img&videos/pi4test.mp4")
bk=cv2.imread('/Users/aidanobrien/Documents/GitHub/BeeFanningDetector/Assets/test_img&videos/pi4test.png')
bk2=cv2.imread('/Users/aidanobrien/Documents/GitHub/BeeFanningDetector/Assets/test_img&videos/black.png')
frames={}
foundbee={}
found=False
def checkWings(c):
    global numfan
    global frames
    global foundbee
    found=False
    mom=cv2.moments(c)
    cx = int(mom["m10"] / mom["m00"])
    cy = int(mom["m01"] / mom["m00"])
    if(frames.get(tuple([cx,cy])) is not None):
        frames[cx,cy]+=1
    else:
        for cY in range (cy-10,cy+10):
            for cX in range (cx-20,cx+20):
                if(frames.get(tuple([cX,cY])) is not None):
                    frames[cX,cY]+=1
                    found=True
                    if(frames[cX,cY]>20 and foundbee.get(tuple([cX,cY]))==False):
                        print("Fanning Detected")
                        foundbee[cX,cY]=True
                        numfan+=1
                    break
        if(found==False):
            frames[cx,cy]=1
            foundbee[cx,cy]=False
bk=bk[100:100+240,0:0+640]
bk2=bk2[100:100+240,0:0+640]
while True:
    hasframes,img=vs.read()
    
    img=img[100:100+240,0:0+640]
    subImage=(bk.astype('int32')-img.astype('int32')).clip(0).astype('uint8')
    grey=cv2.cvtColor(subImage,cv2.COLOR_BGR2GRAY)
    retval,thresh=cv2.threshold(grey,35,255,cv2.THRESH_BINARY)
    #thresh=255-thresh
    kernel=np.ones((5,5),np.uint8)
    thresh=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)
    res2 = cv2.bitwise_and(img, img, mask= thresh)
    

    upper = np.array([255,255,255])  #-- Lower range --
    lower = np.array([128,128,128])  #-- Upper range --
    mask = cv2.inRange(res2, lower, upper)
    res = cv2.bitwise_and(res2, res2, mask= mask)  #-- Contains pixels having the gray color--
    cv2.imshow('Just_Wings/Shadows',res)

    subImage2=(res.astype('int32')-bk2.astype('int32')).clip(0).astype('uint8')
    grey2=cv2.cvtColor(subImage2,cv2.COLOR_BGR2GRAY)
    retval2,thresh2=cv2.threshold(grey2,35,255,cv2.THRESH_BINARY)
    #thresh2=255-thresh2
    kernel2=np.ones((5,5),np.uint8)
    thresh2=cv2.morphologyEx(thresh2,cv2.MORPH_OPEN,kernel)
    im2, contours1, hierarchy1 = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt2=[]
    for c in contours1:
        x,y,w,h=cv2.boundingRect(c)
        if(w*h>150 and w*h < 200 and w > h):
            ell=cv2.fitEllipse(c)
            checkWings(c)
            cv2.ellipse(img,ell,(0,255,0),2)
            #print(w*h)
        else:
            cnt2.append(c)
        
    cv2.drawContours(thresh2, cnt2, -1, (0,0,0), cv2.FILLED)
    cv2.putText(img, "Fanning Bees: {}".format(numfan), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4) 
  
    cv2.imshow('Result',img)

    cv2.imshow('Thresh',thresh2)
    if times > 0:
        key=cv2.waitKey(0) & 0xFF
        #if q is pressed, stop loop
        if key == ord("c"):
            continue
        if key == ord("q"):
            break
    times = times + 1
    