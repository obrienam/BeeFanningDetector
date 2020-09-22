from cv2 import cv2
import numpy as np
from collections import defaultdict

'''
Variables that are important for
this program.
'''
#The number of frames that have past.
times=0

#The number of fanning bees in the current video.
numfan=0

#Video stream used for processing
vs = cv2.VideoCapture("/Users/williebees/Documents/GitHub.nosync/BeeFanningDetector/Assets/test_img&videos/test_vid4.mp4")

#Background image used for initial background subtraction and binary and operations.
bk = cv2.imread('/Users/williebees/Documents/GitHub.nosync/BeeFanningDetector/Assets/test_img&videos/testbkgrd1.jpg')

#Background image used for secont background subtraction and binary and operations. This is used to detect the wings.
bk2 = cv2.imread('/Users/williebees/Documents/GitHub.nosync/BeeFanningDetector/Assets/test_img&videos/black.png')

frames = defaultdict(dict) #Dict for holding the video frames of potentially fanning bees
foundbee = defaultdict(dict) #Dict that holds flags cooresponding to wether or not a fanning bee was found at a particular spot
fanframe = defaultdict(dict) #Dict that holds the most recent frame number when a particular bee was detected.

found = False
sframes = 0
#Only use these cropping bounds in 30 fps vids. This eliminates 
#areas at top/bottom of frame that never contain fanning.
bk=bk[100 : 100 + 240, 0 : 0 + 640]
bk2=bk2[100 : 100 + 240, 0 : 0 + 640]

'''
This function is given a wing contour and
frame image, and determines wether or not 
the cooresponding bee is fanning.
'''
def checkWings(c,img):
    global numfan
    global frames
    global foundbee
    global sframes
    global fanframe
    i = 0
    (x,y),(ma,Ma),angle = cv2.fitEllipse(c)
    x2, y2, w, h = cv2.boundingRect(c)
    ell=cv2.fitEllipse(c)
    found = False
    mom = cv2.moments(c)
    hy = 0
    xw = 0
    #cx and cy tell the  current position on each frame.
    cx = int(mom["m10"] / mom["m00"])
    cy = int(mom["m01"] / mom["m00"])
    if (cy > 100):
        hy = 100
    else:
        hy = 50
    if (cx > 100):
        xw = 100
    else:
        xw=50
    #Checks if the wing has been seen at that (cx, cy) coordinate before.
    if (frames.get(tuple([cx,cy])) is not None and i in frames.get(tuple([cx,cy]))):
        framediff = 0
        if (fanframe.get(tuple([cx,cy])) is not None):
            while framediff < 100:
                if (i in fanframe.get(tuple([cx,cy]))):
                    framediff=sframes-fanframe.get(tuple([cx,cy]))[i]
                else:
                    break

                if framediff >= 100:
                    i += 1
                else:
                    break
        if (i in fanframe.get(tuple([cx,cy]))):
            fanframe[cx,cy][i] = sframes
            frames[cx,cy][i].append(img[cy-hy:cy + hy, cx - xw:cx + xw])
            
    #If the statement gets here then it hasn't found another wing at that same coordinate.
    else:
        #The ranges here check for nearby wing recordings. If it finds one, do the same thing as above.
        for cX in range (cx - 55, cx + 55):
            for cY in range (cy - 10, cy + 10):
                framediff = 0
                if(fanframe.get(tuple([cX,cY])) is not None):
                    while framediff < 100:
                        if(i in fanframe.get(tuple([cX,cY]))):
                            framediff = sframes-fanframe.get(tuple([cX,cY]))[i]
                        else:
                            break
                        if framediff >= 100:
                            i += 1
                        else:
                            break   

                if (frames.get(tuple([cX,cY])) is not None and i in frames.get(tuple([cX,cY]))):
                    if (cY>100):
                        hy=100
                    else:
                        hy=50
                    if (cX>100):
                        xw=100
                    else:
                        xw=50
                    frames[cX,cY][i].append(img[cY-hy:cY+hy,cX-xw:cX+xw])
                    fanframe[cX,cY][i]=sframes
                    found = True
                    #If there are 20 frames or more of this wing and this bee
                    # hasn't already been found, then declare it a fanning bee.
                    if (len(frames.get(tuple([cX,cY]))[i]) >= 20 and foundbee.get(tuple([cX,cY]))[i] == False):
                        print("Fanning Detected")
                        foundbee[cX,cY][i] = True
                        numfan += 1
                    return
        #If the possible wing hasn't been found before, add it to the list of possible wings. 
        if (found == False and cy < 189):
            fanframe[cx,cy][i] = sframes
            frames[cx,cy][i] = [img[cy-hy:cy+hy,cx-xw:cx+xw]]
            foundbee[cx,cy][i] = False


'''
This function iterates through every entry
in the frames dictionary and exports 
videos frames for entries with atleast 
20 frames. Videos can be found in the fanning_exports
directory.
'''
def make_vids():
    i = 0
    global frames
    for key in frames:
        for key2 in frames[key]:
            f = frames[key][key2]
            height, width, layers = f[0].shape
            #Interesting point of note: Sometimes there will be counts of fanning bees
            #at the top without a video exported because their width and height are 0.
            #Should they be counted? What's being counted?
            #To test: use test_vid8.h264 and test_bk7.png
            if (len(f) >= 20 and (width is not 0 and height is not 0)):
                size = (width,height)
                out = cv2.VideoWriter()
                out.open('/Users/williebees/Documents/GitHub.nosync/BeeFanningDetector/Assets/fanning_exports/wings_'+str(key) + ", "+ str(len(f)) + '.mov' ,cv2.VideoWriter_fourcc(*'mp4v'), 10, (size),True)
                for fr in f: 
                    out.write(fr)
                out.release()
                i += 1

def nothing(x):
    pass

'''
This is the main driver loop
that iterates through the provided
video and calls the appropriate functions
to detect fanning bees.
'''
while True:
    sframes += 1
    hasframes,img = vs.read()
    if (hasframes == False):
        break
    
    #Create a pause, uh well it's supposed to be a button but it's a slider.
    switch = '0 : PAUSE \n1 : PLAY'
    

    #Again, in 30 fps top and bottom areas  of the frame are removed.
    img = img[100:100+240,0:0+640]
    subImage = (bk.astype('int32')-img.astype('int32')).clip(0).astype('uint8')
    grey = cv2.cvtColor(subImage,cv2.COLOR_BGR2GRAY)
    retval,thresh = cv2.threshold(grey,35,255,cv2.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)
    thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)
    noback = cv2.bitwise_and(img, img, mask= thresh)
    cv2.imshow('noback',noback)

    #Color Bounds for 30 fps vids
    upper = np.array([255,255,255])  
    lower = np.array([128,128,128])  
    mask = cv2.inRange(noback, lower, upper)

    wings = cv2.bitwise_and(noback, noback, mask=mask)
    
    cv2.imshow('Just_Wings/Shadows',wings)

    subImage2 = (wings.astype('int32')-bk2.astype('int32')).clip(0).astype('uint8')
    grey2 = cv2.cvtColor(subImage2,cv2.COLOR_BGR2GRAY)
    retval2,thresh2 = cv2.threshold(grey2,35,255,cv2.THRESH_BINARY)
    kernel2 = np.ones((5,5),np.uint8)
    thresh2 = cv2.morphologyEx(thresh2,cv2.MORPH_OPEN,kernel)
    im2, contours1, hierarchy1 = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt2 = []
    for c in contours1:
        x, y, w, h = cv2.boundingRect(c)
        r = w / h
        #wing ellipse bounds for 30 fps video 
        if (w * h > 150 and w * h < 200 and w > h):
            ell = cv2.fitEllipse(c)
            checkWings(c,img)
            cv2.ellipse(img,ell,(0,255,0),2)
            
        else:
            cnt2.append(c)
        
    
    cv2.putText(img, "Fanning Bees: {}".format(numfan), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4) 
  
    cv2.imshow('Result',img)
    cv2.createTrackbar(switch, 'Result',1,1,nothing)

    cv2.imshow('Thresh',thresh2)
    if times > 0:
        s = cv2.getTrackbarPos(switch,'Result')
        key = cv2.waitKey(s) & 0xFF
        #Replace waitKey with 0 to go frame by frame, press c to advance.
        #Set waitKey to 1 to play at normal speed.
        #if q is pressed, stop loop
        #Switch will also do this, set to 0 to pause, press c to advance
        #or resume if you set the switch back to 1.
        if key == ord("c"):
            continue
        if key == ord("q"):
            break
    times += 1
vs.release()
cv2.destroyAllWindows()
make_vids()