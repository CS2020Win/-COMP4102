import cv2
import numpy as np

slider_max = 100


def draw_corner_on_image(org, bin_map):
    ret = org.copy()
    w, h = ret.shape
    for i in range(w):
        for j in range(h):
            if bin_map[i][j]:
                ret = cv2.circle(ret, [i, j], 3, [0, 255, 0])
    return ret


def on_trackbar_nms(val):
    dilated = cv2.dilate(minEigenVal, None)
    local_max = cv2.compare(minEigenVal, dilated, cv2.CMP_EQ)
    after_threshold = cv2.threshold(minEigenVal, val / slider_max * 0.02, 255, cv2.THRESH_BINARY)[1]
    after_threshold = np.uint8(after_threshold)

    after_bit_and = cv2.bitwise_and(after_threshold, local_max)
    final_ret = draw_corner_on_image(img, after_bit_and)
    cv2.imshow(windows_title_nms, final_ret)


def on_trackbar_no_nms(val):
    after_bit_and = cv2.threshold(minEigenVal, val / slider_max * 0.02, 255, cv2.THRESH_BINARY)[1]
    final_ret = draw_corner_on_image(img, after_bit_and)
    cv2.imshow(windows_title_no_nms, final_ret)


if __name__ == '__main__':
    img = cv2.imread("box_in_scene.png", cv2.IMREAD_GRAYSCALE)
    cv2.imshow("original Image", img)

    minEigenVal = cv2.cornerMinEigenVal(img, 3)

    windows_title_no_nms = "after_threshold_no_nms"
    taskbar_name = "threshold:"
    cv2.namedWindow(windows_title_no_nms)
    cv2.createTrackbar(taskbar_name, windows_title_no_nms, 0, slider_max, on_trackbar_no_nms)

    on_trackbar_no_nms(50)

    windows_title_nms = "after_threshold_nms"
    cv2.namedWindow(windows_title_nms)
    cv2.createTrackbar(taskbar_name, windows_title_nms, 0, slider_max, on_trackbar_nms)

    on_trackbar_nms(50)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
