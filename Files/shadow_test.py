import cv2
import numpy as np


img = cv2.imread("C:/Users/obrienam/Documents/GitHub/BeeFanningDetector/Assets/test_fan.jpg")
dilated_image=cv2.dilate(img,np.ones((7,7),np.uint8))
bg_img=cv2.medianBlur(dilated_image,21)
diff_img=255-cv2.absdiff(img,bg_img)
norm_img=diff_img.copy()
cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
_, thr_img = cv2.threshold(norm_img, 230, 0, cv2.THRESH_TRUNC)
cv2.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
bk=cv2.imread("C:/Users/obrienam/Documents/GitHub/BeeFanningDetector/Assets/test_bk2.jpg")
thr_img=cv2.absdiff(thr_img,bk)
thr_img=cv2.cvtColor(thr_img, cv2.COLOR_BGR2GRAY)
im2,contours,hierarchy = cv2.findContours(thr_img,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,contours,-1,(0,255,0),3)
cv2.imwrite("C:/Users/obrienam/Documents/GitHub/BeeFanningDetector/Assets/test_shadow.jpg",img)