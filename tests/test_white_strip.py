import cv2
import numpy as np
from typing import Union, Dict

def is_cube_close_to_tape(
    img_or_path: Union[str, np.ndarray],
    close_frac: float = 0.25,   # fraction of tape thickness used as threshold
    min_thr_px: int = 6,        # minimum pixel threshold
    annotate_path: str = None   # if set, save an annotated PNG here
) -> Dict:
    """
    Returns:
      {
        'cube_center': (x, y),
        'tape_center': (x, y),
        'vertical_distance_pixels': int,
        'euclidean_distance_pixels': int,
        'tape_height_pixels': int,
        'threshold_pixels': int,
        'is_close': bool,
        'annotated_image_path': str or None
      }
    """
    # --- Load image ---
    if isinstance(img_or_path, str):
        img_bgr = cv2.imread(img_or_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image at {img_or_path}")
    else:
        img_bgr = img_or_path.copy()
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError("Expected a color image (H,W,3).")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # --- Detect red cube (handle hue wrap-around) ---
    # Tune S/V floors if lighting changes
    lower_red1 = np.array([0,   90,  90], np.uint8)
    upper_red1 = np.array([10, 255, 255], np.uint8)
    lower_red2 = np.array([170, 90,  90], np.uint8)
    upper_red2 = np.array([180, 255, 255], np.uint8)

    mask_r = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, np.ones((5,5), np.uint8), iterations=1)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)

    cnts_r, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts_r:
        raise RuntimeError("No red object found. Adjust red HSV thresholds.")
    cube_cnt = max(cnts_r, key=cv2.contourArea)
    M = cv2.moments(cube_cnt)
    if M['m00'] == 0:
        raise RuntimeError("Degenerate red contour (zero area).")
    cx_cube = int(M['m10'] / M['m00'])
    cy_cube = int(M['m01'] / M['m00'])

    # --- Detect white tape (low saturation, high value) ---
    lower_white = np.array([0,   0, 150], np.uint8)
    upper_white = np.array([179, 60, 255], np.uint8)
    mask_w = cv2.inRange(hsv, lower_white, upper_white)
    mask_w = cv2.morphologyEx(mask_w, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8), iterations=2)

    cnts_w, _ = cv2.findContours(mask_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts_w:
        raise RuntimeError("No white tape found. Adjust white HSV thresholds.")
    tape_cnt = max(cnts_w, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(tape_cnt)
    cx_tape = x + w // 2
    cy_tape = y + h // 2

    # --- Distances & decision ---
    dy = abs(cy_cube - cy_tape)
    euclid = int(np.hypot(cx_cube - cx_tape, cy_cube - cy_tape))
    thr = int(max(min_thr_px, close_frac * h))
    is_close = dy <= thr

    annotated_path = None
    if annotate_path:
        vis = img_rgb.copy()
        cv2.drawContours(vis, [cube_cnt], -1, (0, 255, 0), 2)    # cube outline
        cv2.drawContours(vis, [tape_cnt], -1, (255, 0, 0), 2)    # tape outline
        cv2.circle(vis, (cx_cube, cy_cube), 6, (0, 255, 0), -1)  # cube center
        cv2.circle(vis, (cx_tape, cy_tape), 6, (255, 0, 0), -1)  # tape center
        cv2.line(vis, (cx_cube, cy_cube), (cx_cube, cy_tape), (255, 255, 0), 2)
        label = f"dy={dy}px thr={thr}px -> {'CLOSE' if is_close else 'FAR'}"
        cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        # save as BGR
        cv2.imwrite(annotate_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        annotated_path = annotate_path

    return {
        "cube_center": (cx_cube, cy_cube),
        "tape_center": (cx_tape, cy_tape),
        "vertical_distance_pixels": int(dy),
        "euclidean_distance_pixels": int(euclid),
        "tape_height_pixels": int(h),
        "threshold_pixels": int(thr),
        "is_close": bool(is_close),
        "annotated_image_path": annotated_path,
    }

# --- Example usage ---
if __name__ == "__main__":
    # Replace with your path
    path = "images/test_img_false.jpg"
    result = is_cube_close_to_tape(path, close_frac=0.25, min_thr_px=6, annotate_path="annotated_cube_tape.png")
    print(result)
