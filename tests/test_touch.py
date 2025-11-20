'''
This code is used in the panda push cube task

'''

import cv2
import cv2
import numpy as np

def has_contant(img: np.ndarray) -> bool:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # red wraps around hue, so use two ranges
    lower_red1 = np.array([0, 80, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 80, 80])
    upper_red2 = np.array([180, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    cube_mask = cv2.bitwise_or(mask_red1, mask_red2)

    # clean up
    kernel = np.ones((3, 3), np.uint8)
    cube_mask = cv2.morphologyEx(cube_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    cube_mask = cv2.morphologyEx(cube_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    h, w = img.shape[:2]
    roi = img[int(h*0.6):h, :]  # bottom 40% only

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # broad threshold for “not red, not background wood”
    # tweak these by eye
    lower_gray = np.array([0, 0, 80])
    upper_gray = np.array([180, 60, 255])
    fingers_mask_roi = cv2.inRange(hsv_roi, lower_gray, upper_gray)

    fingers_mask_roi = cv2.morphologyEx(fingers_mask_roi, cv2.MORPH_OPEN, kernel, iterations=2)
    fingers_mask = np.zeros_like(cube_mask)
    fingers_mask[int(h*0.6):h, :] = fingers_mask_roi

    # dilate cube a tiny bit so “almost touching” counts as touching
    dilated_cube = cv2.dilate(cube_mask, kernel, iterations=1)

    overlap = cv2.bitwise_and(dilated_cube, fingers_mask)
    touching = np.any(overlap > 0)
    return touching

img = cv2.imread("images/my_photo-9.jpg")
img = img[:-50, 100:, :]
# --- Red cube mask ---
if has_contant(img):
    print("Touching")
