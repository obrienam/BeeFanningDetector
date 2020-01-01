import cv2

vs=cv2.VideoCapture("/Users/aidanobrien/Documents/GitHub/CV_Research/Assets/bees2.mp4")
firstFrame=None
while True:
    hasFrames,frame=vs.read()
    if (hasFrames==False):
        break
    cv2.imshow("vid",frame)
    if firstFrame is not None:
        print(blah)
    key=cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    firstFrame=frame

vs.release()
cv2.destroyAllWindows()

    