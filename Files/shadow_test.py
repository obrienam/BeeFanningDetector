import cv2
import numpy as np

def to_thresh(img,bk):
    subImage=(bk.astype('int32')-img.astype('int32')).clip(0).astype('uint8')
    grey=cv2.cvtColor(subImage,cv2.COLOR_BGR2GRAY)
    retval,thresh=cv2.threshold(grey,35,255,cv2.THRESH_BINARY_INV)
    thresh=255-thresh
    kernel=np.ones((5,5),np.uint8)
    thresh=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)
    return thresh

img = cv2.imread("C:/Users/obrienam/Documents/GitHub/BeeFanningDetector/Assets/fan_ref/test_frame.png")
bk = cv2.imread("C:/Users/obrienam/Documents/GitHub/BeeFanningDetector/Assets/fan_ref/bk_3_2.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
cv2.imwrite('C:/Users/obrienam/Documents/GitHub/BeeFanningDetector/Assets/fan_ref/fg_test.png', sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
# Marker labelling
ret, markers = cv2.connectedComponents(thresh)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers = cv2.watershed(img,markers)
img[markers == -1] = [0,0,255]
cv2.imwrite('C:/Users/obrienam/Documents/GitHub/BeeFanningDetector/Assets/fan_ref/img_test.png', img)
cv2.imwrite('C:/Users/obrienam/Documents/GitHub/BeeFanningDetector/Assets/fan_ref/bg_test.png', sure_bg)
'''
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
cv2.drawContours(img, contours, 0, (0,255,0), 3)
print(contours[0].size)

#thresh=to_thresh(img,bk)
cv2.imwrite('C:/Users/obrienam/Documents/GitHub/BeeFanningDetector/Assets/fan_ref/thresh_3.jpg', thresh)
cv2.imwrite('C:/Users/obrienam/Documents/GitHub/BeeFanningDetector/Assets/fan_ref/img_cnt_test.jpg', img)
'''