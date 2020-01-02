import cv2
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
        img1=firstFrame
        img2=frame
        bk = cv2.imread('Assets/bee-background.png')
        subImage1=(bk.astype('int32')-img1.astype('int32')).clip(0).astype('uint8')
        grey1=cv2.cvtColor(subImage1,cv2.COLOR_BGR2GRAY)
        retval1,thresh1=cv2.threshold(grey1,35,255,cv2.THRESH_BINARY_INV)
        img1=grey1
        thresh3=thresh1
        im2, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(firstFrame, contours, -1, (0,255,0), 3)
        cv2.imshow("thresh",thresh1)
        cv2.imshow("thresh2",firstFrame)
        
       


        subImage2=(bk.astype('int32')-img2.astype('int32')).clip(0).astype('uint8')
        grey2=cv2.cvtColor(subImage2,cv2.COLOR_BGR2GRAY)
        retval2,thresh2=cv2.threshold(grey2,35,255,cv2.THRESH_BINARY_INV)
        
        img2=grey2
        firstFrame=frame
    else:    
        firstFrame=frame
    key=cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    
    
    cv2.imshow("vid",frame)
vs.release()
cv2.destroyAllWindows()