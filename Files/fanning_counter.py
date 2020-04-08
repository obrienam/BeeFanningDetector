import cv2
import numpy as np
import time
import random as rng
d_frames={}
rects={}
ellipses=[]
times = 0
#convert input frame to threshold using background subtraction
def to_thresh(img,bk):
    subImage=(bk.astype('int32')-img.astype('int32')).clip(0).astype('uint8')
    grey=cv2.cvtColor(subImage,cv2.COLOR_BGR2GRAY)
    retval,thresh=cv2.threshold(grey,35,255,cv2.THRESH_BINARY_INV)
    thresh=255-thresh
    kernel=np.ones((5,5),np.uint8)
    thresh=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)
    return thresh

#Compare the area, central point, shape, and ellipse angle
#of two contours to determine if the stationary bee in the contours is fanning.
def checkFanning(frame,c1,c2):
    
    match=cv2.matchShapes(c1,c2,cv2.CONTOURS_MATCH_I1,0.0)

    #Compute center coordinate of the contours
    mom1=cv2.moments(c1)
    cx1 = int(mom1["m10"] / mom1["m00"])
    cy1 = int(mom1["m01"] / mom1["m00"])
    mom2=cv2.moments(c2)
    cx2 = int(mom2["m10"] / mom2["m00"])
    cy2 = int(mom2["m01"] / mom2["m00"])
    #Compare the center coordinates of the two contours
    cx=cx1/cx2
    cy=cy1/cy2
    #Compare the areas of the two contours
    a1=cv2.contourArea(c1)
    a2=cv2.contourArea(c2)
    a=a1/a2
    detected=False

    #Check if the two contours "match"
    if (cx <= 1.0 and cx >= 0.95 and cy <= 1.0 and cy >= 0.95 
        and a <= 1.0 and a >= 0.95 and a1 < 10000 and match <= 0.3):
        #Draw an ellipse around the contour
        eframe=frame.copy()
        ell=cv2.fitEllipse(c1)
        (x,y),(ma,Ma),angle=cv2.fitEllipse(c1)
        cv2.ellipse(eframe,ell,(0,255,0),2)

        #Check for an existing ellipse near by in the hive. 
        #If found, append frame to dictionary entry at the
        #nearby location.
        for cX in range(cx1-20,cx1+20):
            for cY in range(cy1-55,cy1+55):
                
                if(d_frames.get(tuple([cX,cY])) is not None and 
                #(Ma>=42 and Ma<=49)and ma/Ma>=1.6 and ma/Ma<=2.1 and 
                (angle >125 or angle <90) and cY>100):
                    x,y,w,h=rects.get(tuple([cX,cY]))
                    #cv2.drawContours(frame, c1, -1, (0,255,0), 3)
                    eframe=eframe[y-55:y+h+55,x-20:x+w+20]
                    ellipses.append(ell)
                    d_frames[cX,cY].append(eframe)
                    return True
        #If ellipse isn't found nearby, insert frame to frame dictionary.
        if(d_frames.get(tuple([cx1,cy1])) is None and detected==False and 
        ((ma>=42 and ma<=49 and Ma>=40 and Ma<=90)or(ma>=50 and ma<=63 and Ma>=190 and Ma<=205))and Ma/ma>=1.6 and Ma/ma<=10.5 and (angle > 125 or (angle < 70 and angle > 20)) 
        and cy1>100):
            #print("recognized" + str(cx1))
            x,y,w,h=cv2.boundingRect(c1)
            eframe=eframe[y-55:y+h+55,x-20:x+w+20]
            ellipses.append(ell)  
            d_frames[cx1,cy1]=[eframe]
            rects[cx1,cy1]=[x,y,w,h]
            return True
    
    return False

#remove moving/not fanning bees from threshold frame
#and detect any of the fanning bees frames
def rem_movement(im,thresh,cnt1,cnt2):
    #loop through first conotur list
    numFan=0
    cntmoving=[]
    global ellipses
    ellipses=[]
    imc=im
    numMatch=0
    for c1 in cnt1:
        found=False
        #loop through second contour list
        for c2 in cnt2:
            #check if bee was stationary and was in the 
            #size range of a staionary bee
            imc=im.copy() 
            if c1.size>440 and checkFanning(im,c1,c2):
                found=True
                numMatch+=1
        if(found==False):
            #if not found, append it to moving contours
            cntmoving.append(c1)
    if(numMatch>0 and len(ellipses)>0):
        cframe=im.copy()
        print(len(ellipses))
        for e in ellipses:
            cv2.ellipse(cframe,e,(0,255,0),2)
        cv2.imshow("Ellipse",cframe)
    #fill contours that moved with white
    cv2.drawContours(thresh, cntmoving, -1, (0,0,0), -1)
    return thresh,numFan,imc

#function for exporting fanning bee frames from dictionary
#into seperate videos.
def make_vids(d_frames):
    i = 0
    for key in d_frames:
        frames=d_frames[key]
        height, width, layers = frames[0].shape
        if(len(frames)>15 and (width is not 0 and height is not 0)):
            size = (width,height)
            out = cv2.VideoWriter()
            out.open('/Users/aidanobrien/Documents/GitHub/BeeFanningDetector/Assets/fanning_exports/fan_'+str(key)+", "+str(len(frames))+'.mov',cv2.VideoWriter_fourcc(*'mp4v'), 10, (size),True)
            for f in frames:
                
                out.write(f)
            out.release()
            i=i+1
'''
Experimental watershed function
def wshed(image,bk):
    conts=[]
    shifted=cv2.pyrMeanShiftFiltering(image,21,51)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh=255-thresh
    cv2.imshow("Thresh", thresh)

    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=20,
        labels=thresh)

    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)

    for label in np.unique(labels):
        
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        conts.append(c)
    return conts
'''
#main driver function
def main():
    #windows video file path
    #vs=cv2.VideoCapture("C:/Users/obrienam/Documents/GitHub/BeeFanningDetector/Assets/test_vid2.mp4")
    #mac video file path
    vs=cv2.VideoCapture("/Users/aidanobrien/Documents/GitHub/BeeFanningDetector/Assets/test_img&videos/test_vid4.mp4")
    img1=None

    #loop through video frames 
    while True:
        global times
        hasFrames,img2=vs.read()
        if (hasFrames==False):
            break
        
        if img1 is not None:
            #mac file path
            bk=cv2.imread('/Users/aidanobrien/Documents/GitHub/BeeFanningDetector/Assets/test_img&videos/testbkgrd1.jpg')
            #bk=cv2.imread('C:/Users/obrienam/Documents/GitHub/BeeFanningDetector/Assets/testbkgrd1.jpg')
            
            #crop bk and image frame to appropriate 
            #region of interest
            bk=bk[75:75+260,0:0+640]
            img2=img2[75:75+260,0:0+640]
            

            #take first threshold
            thresh1=to_thresh(img1,bk)
            
            #find first set of frame contours
            im2, contours1, hierarchy1 = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            #cv2.drawContours(img1, contours1, -1, (0,255,0), 3)
            
            #find second threshold
            thresh2=to_thresh(img2,bk)
            #find second set of frame contours
            im3, contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            #remove contours of moving/non-fanning bees and detect fanning bee frames
            thresh1,curFan,imc=rem_movement(img1,thresh1,contours1,contours2)
            

            #display treshold and video frames
            cv2.imshow("threshold",thresh1)
            cv2.imshow("vid_feed",img1)
            
            #increment img2
            img1=img2
        else:    
            img1=img2
            img1=img1[75:75+260,0:0+640]
        if times > 0:
            key=cv2.waitKey(1) & 0xFF
            #if q is pressed, stop loop
            if key == ord("c"):
                continue
            if key == ord("q"):
                break
        times = times + 1
        #time.sleep(0.5)
    #export frames of fanning bees into seperate videos.
    #these are found in assets/fanning_exports
    make_vids(d_frames)
    #close all windows and video stream    
    vs.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()