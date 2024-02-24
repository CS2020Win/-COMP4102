import numpy as np
import cv2


def last_right_pixel_x(img):
    h, w = img.shape
    for j in reversed(range(w)):
        for i in range(h):
            if img[i, j] != 0:
                return j


def first_left_pixel_x(img):
    h, w = img.shape
    for j in range(w):
        for i in range(h):
            if img[i, j] != 0:
                return j


def optimize_seam(left, right, dst):
    cols = last_right_pixel_x(left)  # left_img right border
    o_l = first_left_pixel_x(right)  # right_img left border
    o_w = cols - o_l
    for i in range(left.shape[0]):
        for j in range(o_l, cols):
            if right[i, j] == 0:
                alpha = 1
            elif left[i, j] == 0:
                alpha = 0
            else:
                alpha = (o_w - (j - o_l)) / o_w
            dst[i, j] = left[i, j] * alpha + right[i, j] * (1 - alpha)
    return dst


img1 = cv2.imread('uttower_right.jpg', 0)  # queryImage
img2 = cv2.imread('large2_uttower_left.jpg', 0)  # trainImage

detector = cv2.AKAZE_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = detector.detectAndCompute(img1, None)
kp2, des2 = detector.detectAndCompute(img2, None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1, des2)

# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)

src_pts = np.float32([kp1[matches[m].queryIdx].pt for m in range(0, 20)]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[matches[m].trainIdx].pt for m in range(0, 20)]).reshape(-1, 1, 2)

cv2.imshow('half', img1)
cv2.imshow('long', img2)

t_m, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
warped_img = cv2.warpPerspective(img1, t_m, list(reversed(img2.shape)))
cv2.imshow('warped image', warped_img)
cv2.imwrite("warped.jpg", warped_img)

tmp = cv2.bitwise_or(img2, warped_img)
merged_img = optimize_seam(img2, warped_img, tmp)
cv2.imshow("merged image", merged_img)
cv2.imwrite("merged.jpg", merged_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
