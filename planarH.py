import numpy as np
import cv2

def computeH(x1, x2):
    #Q3.6
    #Compute the homography between two sets of x
    # Construct the A matrix for computing homography
    point_count = x1.shape[0]
    matrix_A = np.zeros((2 * point_count, 9))
    for i in range(point_count):
        x_1, y_1 = x1[i]
        x_2, y_2 = x2[i]
        matrix_A[2 * i] = [-x_2, -y_2, -1, 0, 0, 0, x_2 * x_1, y_2 * x_1, x_1]
        matrix_A[2 * i + 1] = [0, 0, 0, -x_2, -y_2, -1, x_2 * y_1, y_2 * y_1, y_1]
    _, _, Vh = np.linalg.svd(matrix_A)
    H2to1 = Vh[-1].reshape((3, 3))
    return H2to1

def computeH_norm(x1, x2):
    # Normalize x in x1 and x2 before computing homography
    def normalize(x):
        centroid = np.mean(x, axis=0)
        furthest_distance = np.max(np.sqrt(np.sum((x - centroid) ** 2, axis=1)))
        scale_factor = np.sqrt(2) / furthest_distance
        normalization_matrix = np.array([[scale_factor, 0, -scale_factor * centroid[0]],
                                          [0, scale_factor, -scale_factor * centroid[1]],
                                          [0, 0, 1]])
        x_normalized = np.dot(normalization_matrix, np.vstack((x.T, np.ones(x.shape[0]))))
        return x_normalized[:2].T, normalization_matrix

    normalized_x1, transformation1 = normalize(x1)
    normalized_x2, transformation2 = normalize(x2)
    normalized_homography = computeH(normalized_x1, normalized_x2)
    H2to1 = np.dot(np.linalg.inv(transformation1), np.dot(normalized_homography, transformation2))
    return H2to1

def computeH_ransac(x1, x2, num_iterations=70, inlier_threshold=5):
    bestH2to1 = None
    max_inliers_count = 0
    inliers = np.array([])
    for _ in range(num_iterations):
        random_indices = np.random.choice(x1.shape[0], 4, replace=False)
        subset_x1 = x1[random_indices]
        subset_x2 = x2[random_indices]
        trial_homography = computeH_norm(subset_x1, subset_x2)
        x2_homogeneous = np.vstack((x2.T, np.ones(x2.shape[0])))
        projected_x1_homogeneous = np.dot(trial_homography, x2_homogeneous)
        projected_x1 = projected_x1_homogeneous[:2] / projected_x1_homogeneous[2]
        distances = np.linalg.norm(x1 - projected_x1.T, axis=1)
        inliers = np.where(distances < inlier_threshold)[0]
        if len(inliers) > max_inliers_count:
            bestH2to1 = trial_homography
            max_inliers_count = len(inliers)
            inliers = inliers
    return bestH2to1, inliers

def compositeH(H2to1, template, img):
    # Create mask of the same size as template for compositing
    mask = np.ones_like(template)
    target_height, target_width = img.shape[:2]
    # Warp mask and template using the computed homography
    warped_mask = cv2.warpPerspective(mask.swapaxes(0, 1), H2to1, (target_height, target_width)).swapaxes(0, 1)
    inverse_mask = np.logical_not(warped_mask).astype(int)
    warped_template = cv2.warpPerspective(template.swapaxes(0, 1), H2to1, (target_height, target_width)).swapaxes(0, 1)
    # Composite the warped template onto the img using the inverse mask
    composite_image = warped_template + img * inverse_mask
    return composite_image
