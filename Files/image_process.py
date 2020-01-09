import cv2
import numpy as np
import sys
#initialize video stream reader
vs=cv2.VideoCapture("/Users/aidanobrien/Documents/GitHub/BeeFanningDetector/Assets/test_vid1.mp4")
#initialize video stream writer
fourcc=cv2.VideoWriter_fourcc(*'mp4v')
vout=cv2.VideoWriter()
frame_width = int(vs.get(3))
frame_height = int(vs.get(4))
vout.open('out.mp4',fourcc,20,(frame_width,frame_height))

#loop through pixels of each frame, increasing 
#contrast by factor of 2.

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

while True:
    hasFrames,image=vs.read()

    if (hasFrames==False):
        break
    #image=cv2.imread("/Users/aidanobrien/Documents/GitHub/BeeFanningDetector/Assets/testbkgrd1.jpg")
    if image is not None:
        sharpened=unsharp_mask(image)
        vout.write(sharpened)
        #cv2.imwrite("Assets/sharpbackground.jpg",sharpened)
    key=cv2.waitKey(1) & 0xFF
    #press q key to manually quit the script.
    if key == ord("q"):
        break

vout.release()
vs.release()