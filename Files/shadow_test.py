import cv2
import numpy as np

def to_thresh(img,bk):
    subImage=(bk.astype('int32')-img.astype('int32')).clip(0).astype('uint8')
    grey=cv2.cvtColor(subImage,cv2.COLOR_BGR2GRAY)
    retval,thresh=cv2.threshold(grey,35,255,cv2.THRESH_BINARY_INV)
    thresh=255-thresh
    #kernel=np.ones((5,5),np.uint8)
    #thresh=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)
    return thresh

img = cv2.imread("/Users/aidanobrien/Documents/GitHub/BeeFanningDetector/Assets/test_fan.jpg",-1)

rgb_planes = cv2.split(img)

result_planes = []
result_norm_planes = []
for plane in rgb_planes:
    dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(plane, bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    result_planes.append(diff_img)
    result_norm_planes.append(norm_img)

result = cv2.merge(result_planes)
result_norm = cv2.merge(result_norm_planes)

cv2.imwrite('shadows_out.jpg', result)
cv2.imwrite('shadows_out_norm.jpg', result_norm)
img=cv2.imread('/Users/aidanobrien/Documents/GitHub/BeeFanningDetector/Assets/shadows_out_norm.jpg')
bk=cv2.imread('/Users/aidanobrien/Documents/GitHub/BeeFanningDetector/Assets/white_bk.jpg')
thresh=to_thresh(img,bk)
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,255,0), 3)
cv2.imwrite('img_contours.jpg',img)
cv2.imwrite('thresh.jpg', thresh)