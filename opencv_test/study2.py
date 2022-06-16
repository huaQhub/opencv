# 人脸识别（图片、摄像头）
import cv2
 
# 2、训练一组人脸
face_detector = cv2.CascadeClassifier('C:/Users/lcl/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
 
 
# 图片中人脸识别
def Face_Detect_Pic(image):
    # 1、转灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # cv2.imshow("gray", gray)
 

    # 3、检测人脸（用灰度图检测，返回人脸矩形坐标(4个角)）
    faces_rect = face_detector.detectMultiScale(gray, 1.3, 3)
    #                                          灰度图  图像尺寸缩小比例  至少检测次数（若为3，表示一个目标至少检测到3次才是真正目标）
    # print("人脸矩形坐标faces_rect：", faces_rect)
 
    # 4、遍历每个人脸，画出矩形框
    dst = image.copy()
    for x, y, w, h in faces_rect:
        cv2.rectangle(dst, (x, y), (x + w, y + h), (0, 0, 255), 3)  #画出矩形框
 
    # 显示
    # cv2.imshow("dst", dst)
 
    return dst
 
 
# 摄像头中人脸识别
def Face_Detect_Cam():
    # 打开摄像头
    capture = cv2.VideoCapture(0)   #0：本地摄像头    1：外接摄像头
 
    while (True):
        # 1、按帧读取视频
        ret, frame = capture.read()     #frame为每一帧的图像
 
        # 2、左右翻转（否则向左右移动的时候，对象右左移动，反着移）
        frame = cv2.flip(frame, 1)
 
        # 3、对每一帧图像人脸识别
        result = Face_Detect_Pic(frame)

        cv2.imshow('frame',result)

        # q键退出（设置读帧间隔时间）
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    capture.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    # 读取图片
    #img = cv2.imread("Resource/faces.jpg")
   # cv2.imshow("img", img)
 
    #Face_Detect_Pic(img)        #人脸识别（图片）
    Face_Detect_Cam()           #人脸识别（视频）
 
    
 