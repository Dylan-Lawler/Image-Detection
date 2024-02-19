import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection
from helper import plotMatches

def matchPics(I1, I2):
	sigma = 0.15
	#Convert Images to GrayScale
	I1gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
	I2gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

	#Detect Features in Both Images
	locs1 = corner_detection(I1gray, sigma)
	locs2 = corner_detection(I2gray, sigma)

	#Obtain descriptors for the computed feature locations
	desc1, locs1 = computeBrief(I1gray, locs1)
	desc2, locs2 = computeBrief(I2gray, locs2)

	#Match features using the descriptors
	matches = briefMatch(desc1, desc2, 0.8)
	# plotMatches(I1, I2, matches, locs1, locs2)
	return matches, locs1, locs2

# cv_cover = cv2.imread('./data/cv_cover.jpg')
# cv_desk = cv2.imread('./data/cv_desk.png')

# matches, locs1, locs2 = matchPics(cv_cover, cv_desk)

# plotMatches(cv_cover, cv_desk, matches, locs1, locs2)

