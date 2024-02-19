import numpy as np
import cv2
from matchPics import matchPics
from scipy import ndimage
import matplotlib.pyplot as plt

#Q3.5
#Read the image and convert to grayscale, if necessary
cv_cover = cv2.imread('./data/cv_cover.jpg')
count = []

for i in range(36):
    # Rotate Image
    rotation = i * 10
    rotated_image = ndimage.rotate(cv_cover, rotation, reshape=False)
    matches, locs1, locs2 = matchPics(cv_cover, rotated_image)
    count.append(len(matches))


# Compute features, descriptors and Match features


# Update histogram
plt.title('Number of Matches at Different Degrees of Rotation')
plt.xlabel('Rotation (degrees/10)')
plt.ylabel('Number of Mathches')
plt.bar(np.arange(36), count)
plt.show()

