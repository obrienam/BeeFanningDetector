import numpy as np
import argparse
import glob
import cv2 as cv

def auto_canny(image, sigma = 0.33):
    v=np.median(image)
    lower=int(max(0,(1.0-sigma)*v))
    upper=int(min(255,(1.0+sigma)*v))
    edged=cv.Canny(image,lower,upper)
    return edged

ap = argparse.ArgumentParser()
ap.add_argument("-i",'--images',required=True,
    help="path to input dataset of images")
args=vars(ap.parse_args())

for imagePath in glob.glob(args['images'] + "/*.jpg"):
    
    image = cv.imread(imagePath)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (3,3), 0)
    wide = cv.Canny(blurred, 10, 100)
    tight = cv.Canny(blurred, 255, 250)
    auto = auto_canny(blurred)
    nm=str.split(imagePath,'.jpg')

    print(nm[0]+"edge.jpg")
    cv.imwrite(nm[0]+"edge.jpg",wide)
    cv.imshow("Original", image)
    cv.imshow("Edges", np.hstack([wide, tight, auto]))
    cv.waitKey(0)