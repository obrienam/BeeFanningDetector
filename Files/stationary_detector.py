import cv2
import numpy as np
vs=cv2.VideoCapture("/Users/aidanobrien/Documents/GitHub/BeeFanningDetector/Assets/test_vid1.mp4")

img1=None
frame_width = int(vs.get(3))
frame_height = int(vs.get(4))

def rem_movement(thresh,cnt1,cnt2):
    #loop through first conotur list
    for c1 in cnt1:
        found=False
        cntmoving=[]
        #loop through second contour list
        for c2 in cnt2:
            #match the contours
            m=cv2.matchShapes(c1,c2,2,0.0)
            if m==0.0:
                #if match, stop searching
                print("match")
                found=True
                break
        if(found==False):
            print("not found")
            #if not found, append it to moving contours
            cntmoving.append(c1)
    #fill contours that moved with white
    cv2.drawContours(thresh, cntmoving, -1, (255,255,255), -1)
    return thresh
while True:
    hasFrames,img2=vs.read()
    if (hasFrames==False):
        break
    
    if img1 is not None:

        #take first threshold
        bk = cv2.imread('Assets/testbkgrd1.jpg')
        subImage1=(bk.astype('int32')-img1.astype('int32')).clip(0).astype('uint8')
        grey1=cv2.cvtColor(subImage1,cv2.COLOR_BGR2GRAY)
        retval1,thresh1=cv2.threshold(grey1,35,255,cv2.THRESH_BINARY_INV)

        #find first img2 contours
        im2, contours1, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img1, contours1, -1, (0,255,0), 3)
        
        #find second threshold
        subImage2=(bk.astype('int32')-img2.astype('int32')).clip(0).astype('uint8')
        grey2=cv2.cvtColor(subImage2,cv2.COLOR_BGR2GRAY)
        retval2,thresh2=cv2.threshold(grey2,35,255,cv2.THRESH_BINARY_INV)
        
        #find second img2 contours
        im3, contours2, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #remove moved contours
        thresh1=rem_movement(thresh1,contours1,contours2)
        
        #show treshold and video with contours
        cv2.imshow("threshold",thresh1)
        cv2.imshow("contours",img1)

        #increment img2
        img1=img2

    else:    
        img1=img2
    key=cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    
    
    cv2.imshow("vid",img2)
vs.release()
cv2.destroyAllWindows()