import cv2.cv as cv
import numpy as np
import Image

MAX_CORNERS = 500;
# Initialize, load two images from the file system, and
# allocate the images and other structures we will need for
# results.
#
imgA = cv.LoadImage("OpticalFlow0.jpg", cv.CV_LOAD_IMAGE_GRAYSCALE);
imgB = cv.LoadImage("OpticalFlow1.jpg", cv.CV_LOAD_IMAGE_GRAYSCALE);
img_sz = cv.GetSize(imgA);
win_size = 10;
imgC = cv.LoadImage("OpticalFlow1.jpg", cv.CV_LOAD_IMAGE_UNCHANGED);

# The first thing we need to do is get the features
# we want to track.
#
eig_image = cv.CreateImage(img_sz, cv.IPL_DEPTH_32F, 1);
tmp_image = cv.CreateImage(img_sz, cv.IPL_DEPTH_32F, 1);
corner_count = MAX_CORNERS;

cornersA = []
# CvPoint2D32f* cornersA        = new CvPoint2D32f[ MAX_CORNERS ];
# cornersA =cvPointTo32f(MAX_CORNERS)

cornersA = cv.GoodFeaturesToTrack(
    imgA,  # image
    eig_image,  # Temporary floating-point 32-bit image
    tmp_image,  # Another temporary image
    #       cornersA,#number of coners to detect
    corner_count,  # number of coners to detect
    0.01,  # quality level
    5.0,  # minDistace
    useHarris=0,
);
cornerA = cv.FindCornerSubPix(
    imgA,
    cornersA,
    #   corner_count,
    (win_size, win_size),
    (-1, -1),
    (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS, 20, 0.03)
);
# Call the Lucas Kanade algorithm
#
# features_found = [ MAX_CORNERS ];
# feature_errors = [ MAX_CORNERS ];
pyr_sz = (imgA.width + 8, imgB.height / 3);
pyrA = cv.CreateImage(pyr_sz, cv.IPL_DEPTH_32F, 1);
pyrB = cv.CreateImage(pyr_sz, cv.IPL_DEPTH_32F, 1);
cornersB = [];
cornersB, features_found, feature_errors = cv.CalcOpticalFlowPyrLK(
    imgA,
    imgB,
    pyrA,
    pyrB,

    cornersA,

    # corner_count,
    (win_size, win_size),
    5,
    (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS, 20, 0.03),
    0
);
# Now make some image of what we are looking at:
#
for i in range(100):
    if (features_found[i] == 0 or feature_errors[i] > 550):
    # printf("Error is %f/n",feature_errors[i]);
    continue;

    print("Got it");

    p0 = (
        cv.Round(cornersA[i][1]),  # how ot get the (x, y)
        cv.Round(cornersA[i][1])
    )
    p1 = (
        cv.Round(cornersB[i][1]),
        cv.Round(cornersB[i][1])
    )
    cv.Line(imgC, p0, p1, cv.CV_RGB(255, 0, 0), 2);

cv.NamedWindow("ImageA", 0);
cv.NamedWindow("ImageB", 0);
cv.NamedWindow("LKpyr_OpticalFlow", 0);
cv.ShowImage("ImageA", imgA);
cv.ShowImage("ImageB", imgB);
cv.ShowImage("LKpyr_OpticalFlow", imgC);
cv.WaitKey(0);