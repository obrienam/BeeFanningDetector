import cv2
import numpy as np
from scipy import ndimage
times=0
numfan=0
vs=cv2.VideoCapture("/Users/aidanobrien/Documents/GitHub/BeeFanningDetector/Assets/test_img&videos/pi4test.mp4")
bk=cv2.imread('/Users/aidanobrien/Documents/GitHub/BeeFanningDetector/Assets/test_img&videos/pi4test.png')
bk2=cv2.imread('/Users/aidanobrien/Documents/GitHub/BeeFanningDetector/Assets/test_img&videos/black.png')
frames={}
fantot={}
foundbee={}
fanframe={}
found=False
sframes=0
def checkWings(c,img):
    global numfan
    global fantot
    global frames
    global foundbee
    global sframes
    global fanframe
    (x,y),(ma,Ma),angle=cv2.fitEllipse(c)
    ell=cv2.fitEllipse(c)
    found=False
    mom=cv2.moments(c)
    cx = int(mom["m10"] / mom["m00"])
    cy = int(mom["m01"] / mom["m00"])
    if(frames.get(tuple([cx,cy])) is not None):
        fantot[cx,cy]+=1
        frames[cx,cy].append(img[cy-100:cy+100,cx-100:cx+100])
    else:
        for cY in range (cy-10,cy+10):
            for cX in range (cx-55,cx+55):
                framediff=0
                if(fanframe.get(tuple([cX,cY])) is not None):
                    
                    framediff=sframes-fanframe.get(tuple([cX,cY]))
                    
                if(frames.get(tuple([cX,cY])) is not None and framediff<20):
                    fanframe[cX,cY]=sframes
                    fantot[cX,cY]+=1
                    print("{},{}".format(cx,cy))
                    frames[cX,cY].append(img[cY-100:cY+100,cX-100:cX+100])
                    found=True
                    if(fantot[cX,cY]>20 and foundbee.get(tuple([cX,cY]))==False):
                        print("Fanning Detected")
                        print("{}, {}".format(cX,cY))
                        cv2.ellipse(img,ell,(255,0,0),2)
                        foundbee[cX,cY]=True
                        numfan+=1
                        #cv2.imshow('wing',img)
                        #key=cv2.waitKey(1) & 0xFF
                        #if q is pressed, stop loop
                        #if key == ord("c"):
                        #    continue
                    break
                
                if(frames.get(tuple([cX,cY])) is not None and framediff>=20):
                    found=True
                    fanframe[cX,cY+1]=sframes
                    fantot[cX,cY+1]=1
                    frames[cX,cY+1]=[img[cY+1-100:cY+1+100,cX-100:cX+100]]
                    foundbee[cX,cY+1]=False
                    break
                    
                
        if(found==False and cy < 189):
            fantot[cx,cy]=1
            fanframe[cx,cy]=sframes
            print(sframes)
            frames[cx,cy]=[img[cy-100:cy+100,cx-100:cx+100]]
            foundbee[cx,cy]=False
def make_vids():
    i = 0
    global frames
    for key in frames:
        print(key)
        f=frames[key]
        
        height, width, layers = f[0].shape
        
        if(len(f)>=20 and (width is not 0 and height is not 0)):
            size = (width,height)
            out = cv2.VideoWriter()
            out.open('/Users/aidanobrien/Documents/GitHub/BeeFanningDetector/Assets/fanning_exports/wings_'+str(key)+", "+str(len(f))+'.mov',cv2.VideoWriter_fourcc(*'mp4v'), 10, (size),True)
            for fr in f:
                
                out.write(fr)
            out.release()
            i=i+1
bk=bk[100:100+240,0:0+640]
bk2=bk2[100:100+240,0:0+640]
while True:
    sframes+=1
    hasframes,img=vs.read()
    if(hasframes == False):
        break
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
            checkWings(c,img)
            cv2.ellipse(img,ell,(0,255,0),2)
            #print(w*h)
        else:
            cnt2.append(c)
        
    cv2.drawContours(thresh2, cnt2, -1, (0,0,0), cv2.FILLED)
    cv2.putText(img, "Fanning Bees: {}".format(numfan), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4) 
  
    cv2.imshow('Result',img)

    cv2.imshow('Thresh',thresh2)
    if times > 0:
        key=cv2.waitKey(1) & 0xFF
        #if q is pressed, stop loop
        if key == ord("c"):
            continue
        if key == ord("q"):
            break
    
    times = times + 1
vs.release()
cv2.destroyAllWindows()
make_vids()
    