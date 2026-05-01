import cv2
import numpy as np

def get_paper_contour(edges):
    """
    Find the largest 4-point contour in the image, assuming it might be paper.
    """
    cnts, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    paper_cnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            paper_cnt = approx
            break
    return paper_cnt

def order_points(pts):
    """
    Order points for perspective transform
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def perspective_transform(image, pts):
    """
    Apply perspective transform to warp paper straight.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def remove_shadows(img):
    """
    Subtract background illumination to remove shadows.
    """
    dilated_img = cv2.dilate(img, np.ones((7, 7), np.uint8)) 
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(img, bg_img)
    return diff_img

def preprocess_sketch(image_path, target_size=(256, 256)):
    """
    Full CV pipeline for preparing a sketch:
    1. Grayscale
    2. Contour detection and perspective transform
    3. Shadow removal
    4. Adaptive Threshold
    5. Morphological Cleanup
    6. Canny Edge Detection
    7. Normalize to target size
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")

    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur slightly for edge detection of paper
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges_for_paper = cv2.Canny(blurred, 50, 150)

    # 2. Detect paper using contour detection
    paper_cnt = get_paper_contour(edges_for_paper)
    
    # 3. Perspective transform (if paper is detected, otherwise use original)
    if paper_cnt is not None:
        warped_gray = perspective_transform(gray, paper_cnt.reshape(4, 2))
    else:
        warped_gray = gray.copy()

    # 4. Remove shadows
    no_shadow = remove_shadows(warped_gray)

    # 5. Adaptive threshold
    thresh = cv2.adaptiveThreshold(no_shadow, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 6. Morphological cleanup
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # 7. Edge detection (Canny) to refine sketch lines
    edges_final = cv2.Canny(opened, 50, 150)
    
    # 8. Normalize to fixed resolution
    final_sketch = cv2.resize(edges_final, target_size, interpolation=cv2.INTER_AREA)

    return final_sketch

# Example usage (Uncomment to test):
# final_img = preprocess_sketch("path_to_drawn_sketch.jpg")
# cv2.imwrite("clean_sketch.png", final_img)
