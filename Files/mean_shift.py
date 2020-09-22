from cv2 import cv2
import numpy as np

from collections import defaultdict

'''
Variables that are important for
this program.
'''
#The number of frames that have past.
times = 0

#The number of fanning bees in the current video.
numfan = 0

#Video stream used for processing
vs = cv2.VideoCapture("/Users/williebees/Documents/GitHub.nosync/BeeFanningDetector/Assets/test_img&videos/test_vid4.mp4")

#Background image used for initial background subtraction and binary and operations.
bk = cv2.imread('/Users/williebees/Documents/GitHub.nosync/BeeFanningDetector/Assets/test_img&videos/testbkgrd1.jpg')

#Background image used for secont background subtraction and binary and operations. This is used to detect the wings.
bk2 = cv2.imread('/Users/williebees/Documents/GitHub.nosync/BeeFanningDetector/Assets/test_img&videos/black.png')

width, length = (800, 800) #Size declaration for both the 2D arrays below the dictionaries

frames = defaultdict(dict) #Dict for holding the video frames of potentially fanning bees
foundbee = defaultdict(dict) #Dict that holds flags cooresponding to whether or not a fanning bee was found at a particular spot
fanframe = defaultdict(dict) #Dict that holds the most recent frame number when a particular bee was detected.
beeframes = [[0] * width] * length #2d array that holds flags cooresponding to whether or not a bee has both been in that position
                                   #and whether it is pointed towards the hive.
beevids = [[0] * width] *length #Bookkeeping array for video exports

found = False
sframes = 0
#Only use these cropping bounds in 30 fps vids. This eliminates 
#areas at top/bottom of frame that never contain fanning.
bk=bk[100 : 100 + 240, 0 : 0 + 640]
bk2=bk2[100 : 100 + 240, 0 : 0 + 640]

def nothing(x):
    pass

def checkWings(c,img):
    global numfan
    global frames
    global foundbee
    global sframes
    global fanframe
    global beeframes
    i = 0
    (x, y),(ma, Ma), angle = cv2.fitEllipse(c)
    x2, y2, w, h = cv2.boundingRect(c)

    ell = cv2.fitEllipse(c)
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

                if (frames.get(tuple([cX, cY])) is not None and i in frames.get(tuple([cX, cY]))):
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
                    #hasn't already been found, then declare it a fanning bee.
                    if (len(frames.get(tuple([cX,cY]))[i]) >= 20 and foundbee.get(tuple([cX,cY]))[i] == False and 
                            beeframes[cx][cy]):
                        #print(beeframes[cx][cy][i])
                        print("Fanning Detected")
                        foundbee[cX,cY][i] = True
                        numfan += 1
                        beevids[cX][cY] = 1
                    #This is error handling because some cases caused the shadows to be counted as fanning bees later
                    #rather than actually removing it from the pool of possible wings.
                    elif (len(frames.get(tuple([cX,cY]))[i]) >= 20 and foundbee.get(tuple([cX,cY]))[i] == False):
                        foundbee[cX, cY][i] = True
                    return
        #If the possible wing hasn't been found before, add it to the list of possible wings. 
        if (found == False and cy < 189):
            fanframe[cx, cy][i] = sframes
            frames[cx, cy][i] = [img[cy-hy:cy+hy,cx-xw:cx+xw]]
            foundbee[cx, cy][i] = False

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
        keyX = key[0]
        keyY = key[1]
        for key2 in frames[key]:
            f = frames[key][key2]
            height, width, layers = f[0].shape
            if (len(f) >= 20 and (width is not 0 and height is not 0) and beevids[keyX][keyY] == 1):
                size = (width, height)
                out = cv2.VideoWriter()
                out.open('/Users/williebees/Documents/GitHub.nosync/BeeFanningDetector/Assets/fanning_exports/wings_'+str(key) + ", "+ str(len(f)) + '.mov' ,cv2.VideoWriter_fourcc(*'mp4v'), 10, (size),True)
                for fr in f: 
                    out.write(fr)
                out.release()
                i += 1


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
    img = img[100: 100 + 240, 0: 0 + 640]
    subImage = (bk.astype('int32')-img.astype('int32')).clip(0).astype('uint8')
    grey = cv2.cvtColor(subImage,cv2.COLOR_BGR2GRAY)
    retval, thresh = cv2.threshold(grey, 35, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    noback = cv2.bitwise_and(img, img, mask = thresh)
    cv2.imshow('noback', noback)

    #Color Bounds for 30 fps vids
    upper = np.array([255, 255, 255])  
    lower = np.array([128, 128, 128])  
    mask = cv2.inRange(noback, lower, upper)

    wings = cv2.bitwise_and(noback, noback, mask=mask)
    
    cv2.imshow('Just_Wings/Shadows',wings)

    subImage2 = (wings.astype('int32') - bk2.astype('int32')).clip(0).astype('uint8')
    grey2 = cv2.cvtColor(subImage2, cv2.COLOR_BGR2GRAY)
    retval2,thresh2 = cv2.threshold(grey2, 35, 255, cv2.THRESH_BINARY)
    kernel2 = np.ones((5, 5), np.uint8)
    thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN,kernel)
    im2, contours1, hierarchy1 = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt2 = []
    for c in contours1:
        x, y, w, h = cv2.boundingRect(c)
        
        track_window = (x, y, w, h)
        # set up the region of interest for tracking
        roi = img[y: y + h, x: x + w]
        hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0,180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        
        #Bee Rectangle bounds, as well as the meanshift algorithm.
        if (w * h > 1000 and w * h < 1500 and h > w):
            
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            # apply meanshift to get the new location
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)
            # Draw it on image
            x, y, w, h = track_window
            beeframes[x][y] = 1
            cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)

        #wing ellipse bounds for 30 fps video 
        if (w * h > 150 and w * h < 200 and w > h):
            ell = cv2.fitEllipse(c)
            checkWings(c, img)
            cv2.ellipse(img, ell, (0, 255, 0), 2)
            
        else:
            cnt2.append(c)
        
    
    cv2.putText(img, "Fanning Bees: {}".format(numfan), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4) 
  
    cv2.imshow('Result', img)

    cv2.createTrackbar(switch, 'Result', 1, 1, nothing)
    cv2.imshow('Thresh',thresh2)
    if times > 0:
        s = cv2.getTrackbarPos(switch,'Result')
        key = cv2.waitKey(s) & 0xFF
        #Replace waitKey with 0 to go frame by frame, press c to advance.
        #Set waitKey to 1 to play at normal speed.
        #if q is pressed, stop loop.
        #Switch will also do this, set to 0 to pause, press c to advance
        #or resume if you set the switch back to 1 and press c.
        if key == ord("c"):
            continue
        if key == ord("q"):
            break
    times += 1
vs.release()
cv2.destroyAllWindows()
make_vids()