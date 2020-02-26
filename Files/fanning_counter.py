import cv2
import numpy as np
import time
import random as rng
d_frames={}
rects={}
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

#compare current bee contours with fanning bee reference contours
#and find match.
def detect_fanning(c1):
    for i in range(3):
        fname="C:/Users/obrienam/Documents/GitHub/BeeFanningDetector/Assets/fan_ref/thresh_"+str(i+1)+".jpg"
        f=cv2.imread(fname)
       
        f=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
        im3, cnts2, hierarchy2 = cv2.findContours(f, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for c2 in cnts2:
            a1=cv2.contourArea(c1)
            a2=cv2.contourArea(c2)
            #compare the two contour areas
            
            m=cv2.matchShapes(c1,c2,cv2.CONTOURS_MATCH_I1,0.0)
            if m<=0.3:
                a=a1/a2
                if(a<=1.1):
                    return m
                else:
                    continue
    return 1000

#Compare the area, central point, and shape
#of two contours to determine if the bee was stationary.
def cmpContours(frame,c1,c2):
    match=cv2.matchShapes(c1,c2,cv2.CONTOURS_MATCH_I1,0.0)

    mom1=cv2.moments(c1)
    cx1 = int(mom1["m10"] / mom1["m00"])
    cy1 = int(mom1["m01"] / mom1["m00"])
    mom2=cv2.moments(c2)
    cx2 = int(mom2["m10"] / mom2["m00"])
    cy2 = int(mom2["m01"] / mom2["m00"])
    cx=cx1/cx2
    cy=cy1/cy2

    a1=cv2.contourArea(c1)
    a2=cv2.contourArea(c2)
    a=a1/a2
    detected=False

    
    if (cx <= 1.0 and cx >= 0.95 and cy <= 1.0 and cy >= 0.95 
        and a <= 1.0 and a >= 0.95 and a1 < 10000 and match <= 0.3):
        eframe=frame.copy()
        ell=cv2.fitEllipse(c1)
        (x,y),(Ma,ma),angle=cv2.fitEllipse(c1)
        cv2.ellipse(eframe,ell,(0,255,0),2)
        
        
        
        cv2.imshow("ellipse",eframe) 
        for cX in range(cx1-20,cx1+20):
            for cY in range(cy1-20,cy1+20):
                if(d_frames.get(tuple([cX,cY])) is not None and 
                (Ma>=42 and Ma<=49)):
                    x,y,w,h=rects.get(tuple([cX,cY]))
                    #cv2.drawContours(frame, c1, -1, (0,255,0), 3)
                    eframe=eframe[y-20:y+h+20,x-20:x+w+20]
                    print(Ma,ma)
                    
                        
                    d_frames[cX,cY].append(eframe)
                    detected=True

        if(d_frames.get(tuple([cx1,cy1])) is None and detected==False and 
        (Ma>=42 and Ma<=49)):
            #print("recognized" + str(cx1))
            x,y,w,h=cv2.boundingRect(c1)
            eframe=eframe[y-20:y+h+20,x-20:x+w+20]
            
            print(Ma,ma)
              
            d_frames[cx1,cy1]=[eframe]
            
            rects[cx1,cy1]=[x,y,w,h]
        
        return True
    
    return False

#remove moving/not fanning bees from threshold frame
#and count total number of fanning bees
def rem_movement(im,thresh,cnt1,cnt2):
    #loop through first conotur list
    numFan=0
    cntmoving=[]
    imc=im
    for c1 in cnt1:
        found=False
        
        #loop through second contour list
        for c2 in cnt2:
            #check if bee was stationary and was in the 
            #size range of a staionary bee
            imc=im.copy() 
            if c1.size>460 and cmpContours(im,c1,c2):
                found=True
        if(found==False):
            #if not found, append it to moving contours
            cntmoving.append(c1)
    
    #fill contours that moved with white
    cv2.drawContours(thresh, cntmoving, -1, (0,0,0), -1)
    return thresh,numFan,imc

def make_vids(d_frames):
    i = 0
   
    for key in d_frames:
        frames=d_frames[key]
        height, width, layers = frames[0].shape
        if(len(frames)>1 and (width is not 0 and height is not 0)):
            
           
            size = (width,height)
            out = cv2.VideoWriter('../Assets/stationary_bees/fan_'+str(key)+", "+str(len(frames))+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (size))
            for f in frames:
                out.write(f)
            out.release()
            i=i+1

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
#main driver function
def main():
    #windows video file path
    vs=cv2.VideoCapture("C:/Users/obrienam/Documents/GitHub/BeeFanningDetector/Assets/test_vid2.mp4")
    #mac video file path
    #vs=cv2.VideoCapture("/Users/aidanobrien/Documents/GitHub/BeeFanningDetector/Assets/test_vid1.mp4")

    img1=None
   
    

    #loop through video frames 

    while True:
        global times
        hasFrames,img2=vs.read()
        if (hasFrames==False):
            break
        #img2=cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)
        
        
        if img1 is not None:
            #mac file path
            #bk=cv2.imread('/Users/aidanobrien/Documents/GitHub/BeeFanningDetector/Assets/testbkgrd1.jpg')
           
            bk=cv2.imread('C:/Users/obrienam/Documents/GitHub/BeeFanningDetector/Assets/testbkgrd1.jpg')
            
            
            bk=bk[75:75+315,0:0+637]
            img2=img2[75:75+315,0:0+637]
            

            #take first threshold
            thresh1=to_thresh(img1,bk)
            
            #find first set of frame contours
            im2, contours1, hierarchy1 = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            #cv2.drawContours(img1, contours1, -1, (0,255,0), 3)
            
            #find second threshold
            thresh2=to_thresh(img2,bk)
            #find second set of frame contours
            im3, contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            #remove contours of moving/non-fanning bees and count fanning bees
            thresh1,curFan,imc=rem_movement(img1,thresh1,contours1,contours2)
            

            #show treshold and video
            cv2.imshow("threshold",thresh1)
            #write current number of fanning bees to current frame
            cv2.putText(img1, "Total Fanning Bees Detected: {}".format(len(d_frames)), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            #draw every contour on the current frame for testing purposes
            #cv2.drawContours(img1, contours1, -1, (0,255,0), 2)
            cv2.imshow("contours",img1)
            
            #increment img2
            img1=img2
        else:    
            img1=img2
            img1=img1[75:75+315,0:0+637]
        if times > 0:
            key=cv2.waitKey(1) & 0xFF
            #if q is pressed, stop loop
            if key == ord("c"):
                continue
            if key == ord("q"):
                break
        times = times + 1
        #time.sleep(1)
    make_vids(d_frames)
    #close all windows and video stream    
    vs.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()