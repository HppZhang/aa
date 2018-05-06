# -*-coding: utf-8 -*-
import cv2
import numpy as np
capture = cv2.VideoCapture("/home/zhp/PycharmProjects/human_kth/runing/person01_running_d4_uncomp.avi")
i = 1
#cv2.namedWindow("human", 0)
#cv2.namedWindow("human_gray_guass", 0)
cv2.namedWindow("human_sobel", 0)
#cv2.namedWindow("dalidate", 0)
#cv2.namedWindow("erode", 0)
while(capture.isOpened()):
    read_flag, frame = capture.read()
    vid_frames = []
    while read_flag:
        read_flag, frame = capture.read()
        vid_frames.append(frame)
        i += 1
    else:
        break
   # vid_frames = np.asarray(vid_frames, dtype='uint8')[:-1]
print "vid_frames"
print i
gray_out = []
gray_out1 = []
gray_out3 = []
gray_diff3 = []
fi_out = []
gray_end = []
for j in range(len(vid_frames)-1):
    gray_out.append(cv2.cvtColor(vid_frames[j], cv2.COLOR_BGR2GRAY))
for jjj in range(len(gray_out)):
    gray_out1.append(cv2.GaussianBlur(gray_out[jjj], (5, 5), 0))
    # cv2.imshow("human_gray", gray_out1[jjj])
for jij in range(len(gray_out1)):
    x = cv2.Sobel(gray_out1[jij], cv2.CV_16S, 1, 0)
    y = cv2.Sobel(gray_out1[jij], cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    #fg = cv2.createBackgroundSubtractorMOG2()
    #dst = fg.apply(dst)
    for i_row in range(dst.shape[0]):
        for i_col in range(dst.shape[1]):
            if abs(dst[i_row][i_col]) >= 68:
                dst[i_row][i_col] = 255
            else:
                dst[i_row][i_col] = 0
    gray_out3.append(dst)
    kernel = np.ones((3, 3), np.uint8)
    gray_diff3.append(gray_out3[jij])
    if gray_diff3[jij].any():
        # cv2.imwrite("/home/zhp/PycharmProjects/human_kth/human" + str(jij) + ".jpg", gray_diff3[jij])
        # gray_end.append(gray_diff3[jij])
        alex = []
        #gray_end_array = np.array(gray_end)
        for i_row_out in range(np.array(gray_diff3[jij]).shape[0]):
            for i_col_out in range(np.array(gray_diff3[jij]).shape[1]):
                if gray_diff3[jij][i_row_out][i_col_out] != 0:
                    alex.append([i_row_out, i_col_out])
        a0 = [alex[0][0], alex[0][1]]  # 定图像最高点的坐标
        a1 = [alex[0][0], alex[0][1]]  # 定图像最左侧点的坐标
        a2 = [alex[0][0], alex[0][1]]  # 定图像最底侧的坐标
        a3 = [alex[0][0], alex[0][1]]  # 定图像最右侧点的坐标
        for alex_out in range(len(alex)):
            if alex[alex_out][0] <= a0[0]:
                a0 = [alex[alex_out][0], alex[alex_out][1]]
            if alex[alex_out][1] <= a1[1]:
                a1 = [alex[alex_out][0], alex[alex_out][1]]
            if alex[alex_out][0] > a2[0]:
                a2 = [alex[alex_out][0], alex[alex_out][1]]
            if alex[alex_out][1] > a3[1]:
                a3 = [alex[alex_out][0], alex[alex_out][1]]
        goal_High = a2[0] - a0[0]
        goal_Weigh = a3[1] - a1[1]
        left_up = (a1[1], a0[0])
        print "left_up"
        print left_up
        right_bottom = (a3[1], a2[0])
        print "right_bottom"
        print right_bottom
        print "gray_diff3[jij].shape"
        print np.array(gray_diff3[jij]).shape
        if right_bottom[0] <= np.array(gray_diff3[jij]).shape[1] and goal_Weigh >= 30  or left_up[0] >= 0 and goal_Weigh >= 30:
            # fi_out.append(gray_diff3[jij][left_up[1]:right_bottom[1]][left_up[0]:right_bottom[0]])
            #cv2.imwrite("/home/zhp/PycharmProjects/human_kth/human" + str(jij) + ".jpg", gray_diff3[jij][left_up[1]:right_bottom[1], left_up[0]:right_bottom[0]])
            fi_out.append(gray_out1[jij][left_up[1]:right_bottom[1], left_up[0]:right_bottom[0]])
            cv2.imshow("human_sobel", gray_out1[jij][left_up[1]:right_bottom[1], left_up[0]:right_bottom[0]])
            if cv2.waitKey(30) & 0xFF == 27:
                break
            #cv2.rectangle(gray_diff3[jij], left_up, right_bottom, (255, 0, 0), 1)
            #cv2.imshow("human", gray_diff3[jij])
            #if cv2.waitKey(30) & 0xFF == 27:
            #    break
    #cv2.imshow("human_sobel", gray_out3[jij])
    #cv2.waitKey()
for qqi in range(len(fi_out)):
     cv2.imwrite("/home/zhp/PycharmProjects/human_kth/run_train/human" + str(qqi) + ".jpg", fi_out[qqi])

'''if cv2.waitKey(30) & 0xFF == 27:
        break'''
'''gray_end_array = np.array(gray_end)
print "gray_end_array_shape"
print gray_end_array.shape
alex = []
for i_out in range(gray_end_array.shape[0]):
    for i_row_out in range(gray_end_array.shape[1]):
        for i_col_out in range(gray_end_array.shape[2]):
            if gray_end_array[i_out][i_row_out][i_col_out] != 0:
                alex.append(i_row_out, i_col_out)
a0 = [alex[0][0], alex[0][1]]  #定图像最高点的坐标
a1 = [alex[0][0], alex[0][1]]  #定图像最左侧点的坐标
a2 = [alex[0][0], alex[0][1]]  #定图像最底侧的坐标
a3 = [alex[0][0], alex[0][1]]  #定图像最右侧点的坐标
for alex_out in range(len(alex)):
    if alex[alex_out][0] < a0[0]:
        a0 = [alex[alex_out][0], alex[alex_out][1]]
    if alex[alex_out][1] < a1[1]:
        a1 = [alex[alex_out][0], alex[alex_out][1]]
    if alex[alex_out][0] > a2[1]:
        a2 = [alex[alex_out][0], alex[alex_out][1]]
    if alex[alex_out][1] > a3[1]:
        a3 = [alex[alex_out][0], alex[alex_out][1]]
goal_High = a2[0] - a0[0]
goal_Weigh = a3[1] - a1[1]
cv2.rectangle()'''
'''for ii in range(len(gray_out3)-2):
    gray_diff1 = np.subtract(gray_out3[ii+1], gray_out3[ii])
    # gray_diff1 = gray_diff1+gray_diff1+gray_diff1
   # cv2.imshow("human", gray_diff1)
   # cv2.waitKey()
    gray_diff2 = np.subtract(gray_out3[ii+2], gray_out3[ii+1])
    # gray_diff2 = gray_diff2+gray_diff2+gray_diff2
   # cv2.imshow("human_gray_guass", gray_diff2)
   # cv2.waitKey()
    for i_row in range(gray_diff1.shape[0]):
        for i_col in range(gray_diff1.shape[1]):
            if abs(gray_diff1[i_row][i_col]) >= 30:
                gray_diff1[i_row][i_col] = 255
            else:
                gray_diff1[i_row][i_col] = 0
    for j_row in range(gray_diff2.shape[0]):
        for j_col in range(gray_diff2.shape[1]):
            if abs(gray_diff2[j_row][j_col]) >= 30: 
                gray_diff2[j_row][j_col] = 255
            else:
                gray_diff2[j_row][j_col] = 0
    gray_diff3 = cv2.bitwise_and(gray_diff1, gray_diff2)
    kernel = np.ones((5, 5), np.uint8)
    gray_diff3 = cv2.erode(gray_diff3, kernel)
    cv2.imshow("erode", gray_diff3)
    if cv2.waitKey(30) & 0xFF == 27:
        break
    gray_diff3 = cv2.dilate(gray_diff3, kernel)
    cv2.imshow("human", gray_diff3)
    if cv2.waitKey(30) & 0xFF == 27:
        break
    cv2.imshow("erode", gray_diff3)
    if cv2.waitKey(30) & 0xFF == 27:
        break
    print "gray_diff1"
    print gray_diff1.shape'''
'''for i1 in range(len(gray_out)):
    # cv2.imwrite("/home/zhp/PycharmProjects/human_kth/human"+str(i1)+".jpg", gray_out[i1])
    cv2.imshow("human", gray_out[i1])
    if cv2.waitKey(30) & 0xFF == 27:
        break'''
'''cv2.waitKey()'''
capture.release()
cv2.destroyAllWindows()
#print len(vid_frames)
'''subtract(gray_out[j1+1], gray_out[j1], gray_diff1); // 第二帧减第一帧
    subtract(gray_out[j1+2], gray_out[j1+1], gray_diff2); // 第三帧减第二帧

    for (int i=0;i < gray_diff1.rows;i++)
    for (int j=0;j < gray_diff1.cols;j++)
    {
    if (abs(gray_diff1.at < unsigned
    char > (i, j)) >= threshold_diff1) // 这里模板参数一定要用unsigned
    char，否则就一直报错
    gray_diff1.at < unsigned
    char > (i, j) = 255; // 第一次相减阈值处理
    else gray_diff1.at < unsigned
    char > (i, j) = 0;

    if (abs(gray_diff2.at < unsigned char > (i, j)) >= threshold_diff2) // 第二次相减阈值处理
    gray_diff2.at < unsigned char > (i, j)=255;
    else gray_diff2.at < unsigned char > (i, j)=0;
    }'''
