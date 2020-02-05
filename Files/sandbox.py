import cv2
import numpy as np
import imutils
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
image = cv2.imread("C:/Users/obrienam/Documents/GitHub/BeeFanningDetector/Assets/fan_ref/test_frame.png")
bk = cv2.imread("C:/Users/obrienam/Documents/GitHub/BeeFanningDetector/Assets/fan_ref/bk_3_2.jpg")

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
    cv2.drawContours(image,[c],0,(0,255,0),1)

cv2.imshow("Output", image)
cv2.waitKey(0)