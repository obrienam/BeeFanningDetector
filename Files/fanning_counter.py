import cv2
import numpy as np
import time

#convert input frame to threshold using background subtraction
def to_thresh(img,bk):
    subImage=(bk.astype('int32')-img.astype('int32')).clip(0).astype('uint8')
    grey=cv2.cvtColor(subImage,cv2.COLOR_BGR2GRAY)
    retval,thresh=cv2.threshold(grey,35,255,cv2.THRESH_BINARY_INV)
    thresh=255-thresh
    kernel=np.ones((5,5),np.uint8)
    thresh=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)
    return thresh

#remove moving/not fanning bees from threshold frame
#and count total number of fanning bees
def rem_movement(thresh,cnt1,cnt2,cnt_fan):
    #loop through first conotur list
    numFan=0
    cntmoving=[]
    for c1 in cnt1:
        found=False
        
        #loop through second contour list
        for c2 in cnt2:
            
            #match the contours
            m=cv2.matchShapes(c1,c2,cv2.CONTOURS_MATCH_I1,0.0)
            if c1.size>460 and c1.size<480 and m <= 0.01:
                #if match, stop searching
                m2=cv2.matchShapes(c1,cnt_fan[0],cv2.CONTOURS_MATCH_I1,0.0)
                if m2 <= 0.3:
                    print("fan")
                    numFan=numFan+1
                else:
                    cntmoving.append(c1)
                found=True
                break
        if(found==False):
            #if not found, append it to moving contours
            cntmoving.append(c1)
    #fill contours that moved with white
    cv2.drawContours(thresh, cntmoving, -1, (0,0,0), -1)
    return thresh,numFan

#main driver function
def main():
    #windows video file path
    #vs=cv2.VideoCapture("C:/Users/obrienam/Documents/GitHub/BeeFanningDetector/Assets/test_vid2.mp4")
    #mac video file path
    vs=cv2.VideoCapture("/Users/aidanobrien/Documents/GitHub/BeeFanningDetector/Assets/test_vid1.mp4")

    img1=None
    frame_width = int(vs.get(3))
    frame_height = int(vs.get(4))

    #initialize fanning bee reference contour
    fan=cv2.imread("/Users/aidanobrien/Documents/GitHub/BeeFanningDetector/Assets/test_fan.jpg")
    b_fan=cv2.imread("/Users/aidanobrien/Documents/GitHub/BeeFanningDetector/Assets/test_bkgrnd.jpg")
    thresh_fan=to_thresh(fan,b_fan)
    im, cnt_fan, hier_fan = cv2.findContours(thresh_fan, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)    

    #loop through video frames 
    while True:
        hasFrames,img2=vs.read()
        if (hasFrames==False):
            break
        
        if img1 is not None:
            #mac file path
            bk=cv2.imread('/Users/aidanobrien/Documents/GitHub/BeeFanningDetector/Assets/testbkgrd1.jpg')
            #bk = rem_shadows(cv2.imread('Assets/sharpbackground.jpg'))
            bk=cv2.imread('/Users/aidanobrien/Documents/GitHub/BeeFanningDetector/Assets/testbkgrd1.jpg')
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
            thresh1,curFan=rem_movement(thresh1,contours1,contours2,cnt_fan)
            #show treshold and video
            cv2.imshow("threshold",thresh1)
            #write current number of fanning bees to current frame
            cv2.putText(img1, "Fanning Bees: {}".format(curFan), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            #draw every contour on the current frame for testing purposes
            #cv2.drawContours(img1, contours1, -1, (0,255,0), 2)
            cv2.imshow("contours",img1)

            #increment img2
            img1=img2
        else:    
            img1=img2
        key=cv2.waitKey(1) & 0xFF
        #if q is pressed, stop loop
        if key == ord("q"):
            break
    #close all windows and video stream    
    vs.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()