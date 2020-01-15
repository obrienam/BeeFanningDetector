import cv2
import numpy as np
import time
#windows file path
vs=cv2.VideoCapture("C:/Users/obrienam/Documents/GitHub/BeeFanningDetector/Assets/12-51-01.mp4")
#mac file path
#vs=cv2.VideoCapture("/Users/aidanobrien/Documents/GitHub/BeeFanningDetector/Assets/test_vid1.mp4")

img1=None
frame_width = int(vs.get(3))
frame_height = int(vs.get(4))

def rem_movement(thresh,cnt1,cnt2):
    #loop through first conotur list
  
    cntmoving=[]
    for c1 in cnt1:
        found=False
        
        #loop through second contour list
        for c2 in cnt2:
            #match the contours
            m=cv2.matchShapes(c1,c2,cv2.CONTOURS_MATCH_I1,0.0)
            if c1.size>150 and m <= 0.01:
                #if match, stop searching
                print(c1.size)
                found=True
                break
        if(found==False):
            #if not found, append it to moving contours
            cntmoving.append(c1)
    #fill contours that moved with white
    cv2.drawContours(thresh, cntmoving, -1, (0,0,0), -1)
    return thresh

def rem_shadows(img):
    rgb_planes = cv2.split(img)
    result_norm_planes=[]
    result_planes=[]
    for plane in rgb_planes:
        dilated_img=cv2.dilate(plane,np.ones((7,7),np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img=255-cv2.absdiff(plane,bg_img)
        norm_img=cv2.normalize(diff_img,None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
        result_planes.append(diff_img)
    result=cv2.merge(result_planes)
    return result
    
bk=cv2.imread('C:/Users/obrienam/Documents/GitHub/BeeFanningDetector/Assets/testbkgrd1.jpg')
while True:
    hasFrames,img2=vs.read()
    if (hasFrames==False):
        break
    
    if img1 is not None:

        #take first threshold
        #windows file path
        
        #mac file path

        #bk = rem_shadows(cv2.imread('Assets/sharpbackground.jpg'))
    
        subImage1=(bk.astype('int32')-img1.astype('int32')).clip(0).astype('uint8')
        grey1=cv2.cvtColor(subImage1,cv2.COLOR_BGR2GRAY)
        retval1,thresh1=cv2.threshold(grey1,35,255,cv2.THRESH_BINARY_INV)
        thresh1=255-thresh1
        kernel=np.ones((5,5),np.uint8)
        thresh1=cv2.morphologyEx(thresh1,cv2.MORPH_OPEN,kernel)
        #find first img2 contours
        im2, contours1, hierarchy1 = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #cv2.drawContours(img1, contours1, -1, (0,255,0), 3)
        
        #find second threshold
        subImage2=(bk.astype('int32')-img2.astype('int32')).clip(0).astype('uint8')
        grey2=cv2.cvtColor(subImage2,cv2.COLOR_BGR2GRAY)
        retval2,thresh2=cv2.threshold(grey2,35,255,cv2.THRESH_BINARY_INV)
        thresh2=255-thresh2
        thresh2=cv2.morphologyEx(thresh2,cv2.MORPH_OPEN,kernel)
        #find second img2 contours
        im3, contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        #remove moved contours
        
        thresh1=rem_movement(thresh1,contours1,contours2)
        cv2.drawContours(img1, contours1, -1, (0,255,0), 2)
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