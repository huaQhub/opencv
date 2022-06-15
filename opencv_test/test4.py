#在 OpenCV 里使用 Camshift
import numpy as np
import cv2 as cv

cap = cv.VideoCapture('202206151435.mp4')

# 获取视频的第一帧
ret,frame = cap.read()

# 设置窗口的初始位置
r,h,c,w = 250,90,400,125  # 简单地硬编码值
track_window = (c,r,w,h)

# 设置 ROI(图像范围)以进行跟踪
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

# 设置结束标志，10 次迭代或至少 1 次移动
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret ,frame = cap.read()

    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # 运行 Camshift 用以获取新的位置
        ret, track_window = cv.CamShift(dst, track_window, term_crit)

        # 绘制到新图像中
        pts = cv.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv.polylines(frame,[pts],True, 255,2)
        cv.imshow('img2',img2)

        k = cv.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv.imwrite(chr(k)+".jpg",img2)
    else:
        break
cv.destroyAllWindows()
cap.release()
