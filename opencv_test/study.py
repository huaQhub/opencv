# 3BackgroundSubtractorMOG
import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)


face_cascade = cv.CascadeClassifier('C:/Users/lcl/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('C:/Users/lcl/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_eye.xml')


def find_face(img, gray):
    faces = face_cascade.detectMultiScale(gray, 1.3, 9)
    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    return img
    # cv.imshow('img',img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


while(1):
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    cv.imshow('frame',cv.flip(find_face(frame,gray),1))

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()

