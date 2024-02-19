import numpy as np
import cv2
import skimage.io 
import skimage.color
#Import necessary functions
from matchPics import *
from planarH import *
from helper import *
import matplotlib.pyplot as plt

# Write script for Q3.9
cv_cover = cv2.imread('./data/cv_cover.jpg')
cv_desk = cv2.imread('./data/cv_desk.png')
hp_cover = cv2.imread('./data/hp_cover.jpg')

# Get matches and compute homography
matches, locs1, locs2 = matchPics(cv_desk, cv_cover)
x1 = locs1[matches.T[0]]
x2 = locs2[matches.T[1]]
H2to1, inliers = computeH_ransac(x1, x2)

# Resize Harry Potter cover to CV Cover
h, w = cv_cover.shape[0], cv_cover.shape[1]
resized_hp_cover = cv2.resize(hp_cover, (w, h))

composite = compositeH(H2to1, resized_hp_cover, cv_desk)
cv2.imwrite("final.png", composite)