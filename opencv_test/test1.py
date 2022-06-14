#蛮力匹配
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img1 = cv.imread('box.png',0)          # queryImage
img2 = cv.imread('box_in_scene.png',0) # trainImage

print("import img success")

# Initiate SIFT detector
sift = cv.SIFT_create()
print("create sift success")
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
print("create kp des success")
# BFMatcher with default params 
bf = cv.BFMatcher()
print("finish bf")
matches = bf.knnMatch(des1,des2, k=2)
print("finish matches")
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
plt.imshow(img3),plt.show()
cv.imwrite('outputimg.png',img3)