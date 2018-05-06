# -*-coding: utf-8 -*-
import cv2
import numpy as np
'''capture = cv2.VideoCapture("/home/zhp/KTH/running/person01_running_d4_uncomp.avi")
i = 1
while(capture.isOpened()):
    read_flag, frame = capture.read()
    vid_frames = []
    while read_flag:
        read_flag, frame = capture.read()
        vid_frames.append(frame)
        i += 1
    else:
        break
gray_out = []
gray_out1 = []
gray_out3 = []
gray_diff3 = []
fi_out = []
gray_end = []
for j in range(len(vid_frames)-1):
    gray_out.append(cv2.cvtColor(vid_frames[j], cv2.COLOR_BGR2GRAY))
                #cv2.imshow("huidu", gray_out[j])
                #cv2.waitKey()
for jjj in range(len(gray_out)):
    gray_out1.append(cv2.GaussianBlur(gray_out[jjj], (5, 5), 0))
                #cv2.imshow("Gaussian", gray_out1[jjj])
                #cv2.waitKey()'''

def detect_video(video):
    gray_out = []
    j = 0
    camera = cv2.VideoCapture(video)
    history = 150    # 训练帧数
    bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)  # 背景减除器，设置阴影检测
    bs.setHistory(history)
    frames = 0
    while True:
        res, frame = camera.read()
        if not res:
            break
        fg_mask = bs.apply(frame)   # 获取 foreground mask
        if frames < history:
            frames += 1
            continue
        # 对原始帧进行膨胀去噪
        cv2.imshow("fg_mask.copy()", fg_mask.copy())
        th = cv2.threshold(fg_mask.copy(), 254, 255, cv2.THRESH_BINARY)[1]
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7)), iterations=2)        # 获取所有检测框
        image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            # 获取矩形框边界坐标
            x, y, w, h = cv2.boundingRect(c)
            # 计算矩形框的面积
            area = cv2.contourArea(c)

            if 1000 < area < 3000:
                gray_out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow("detection", frame)
                cv2.imshow("back", dilated)
                print x,y,w,h
                cv2.imwrite("/home/zhp/PycharmProjects/human_kth/KNN back ground/human" + str(j) + ".jpg", gray_out[j][y:y + h,x:x + w])
                j = j + 1
            #cv2.waitKey()
                if cv2.waitKey(110) & 0xff == 27:
                    break
    camera.release()
if __name__ == '__main__':
    video = '/home/zhp/KTH/running/person01_running_d4_uncomp.avi'
    detect_video(video)
'''函数为cv2.threshold()
这个函数有四个参数，第一个原图像，第二个进行分类的阈值，第三个是高于（低于）阈值时赋予的新值，第四个是一个方法选择参数，常用的有：
• cv2.THRESH_BINARY（黑白二值）
• cv2.THRESH_BINARY_INV（黑白二值反转）
• cv2.THRESH_TRUNC （得到的图像为多像素值）
• cv2.THRESH_TOZERO
• cv2.THRESH_TOZERO_INV
该函数有两个返回值，第一个retVal（得到的阈值值（在后面一个方法中会用到）），第二个就是阈值化后的图像。'''