# BackgroundSubtractorMOG2
import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)#0为电脑内置摄像头

fgbg = cv.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()
    frame = cv.flip(frame,1)
    fgmask = fgbg.apply(frame)

    cv.imshow('frame',fgmask)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()
