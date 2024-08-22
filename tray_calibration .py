import os
import cv2
import numpy as np
from scipy.spatial import distance as dist
from scipy import ndimage

def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    
    return np.array([tl, bl, br, tr], dtype="float32")

def get_transform(p):
    w1 = np.linalg.norm(p[0] - p[3])
    w2 = np.linalg.norm(p[1] - p[2])
    h1 = np.linalg.norm(p[0] - p[1])
    h2 = np.linalg.norm(p[2] - p[3])
    
    wmax = max(int(w1), int(w2))
    hmax = max(int(h1), int(h2))
    
    i = np.float32([p[0], p[1], p[2], p[3]])
    o = np.float32([[0, 0], [0, hmax - 1], [wmax - 1, hmax - 1], [wmax - 1, 0]])
                        
    M = cv2.getPerspectiveTransform(i, o)
    
    return M, hmax, wmax

def process_image(img_path, output_folder, thr='gray'):
    img1 = cv2.imread(img_path)

    if img1 is None:
        raise FileNotFoundError(f"Image at path {img_path} not found.")

    res = img1.shape[0] / 4000

    if thr == 'gray':
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(int(200*res), int(200*res)))
        gray_equalised = clahe.apply(gray)
        ret, thresh = cv2.threshold(gray_equalised, 0, 255, cv2.THRESH_OTSU)
    else:
        HSV = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(int(200*res), int(200*res)))
        HSV_equalised = clahe.apply(HSV[:, :, 1])
        ret, thresh = cv2.threshold(HSV_equalised, 0, 255, cv2.THRESH_OTSU)

    kernel_size = int(res * 5)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((kernel_size * 10, kernel_size * 10), np.uint8))
    mask = ndimage.binary_fill_holes(mask).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("No contours found in the image.")
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    for epsilon_factor in [0.02, 0.04, 0.06]:
        epsilon = epsilon_factor * cv2.arcLength(largest_contour, True)
        poly = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(poly) == 4:
            break
    
    if len(poly) != 4:
        poly = cv2.convexHull(largest_contour)
        if len(poly) > 4:
            poly = cv2.approxPolyDP(poly, epsilon, True)
        if len(poly) != 4:
            x, y, w, h = cv2.boundingRect(largest_contour)
            poly = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype="float32")
    
    poly = poly.reshape((4, 2))
    poly_ordered = order_points(poly)

    M, hmax, wmax = get_transform(poly_ordered)
    img2 = cv2.warpPerspective(img1, M, (wmax, hmax), flags=cv2.INTER_LINEAR)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, os.path.basename(img_path))
    cv2.imwrite(output_path, img2)
    print(f"Processed image saved to {output_path}")

    # Return the processed image and the four coordinates
    return img2, poly_ordered

# Run the process and get the image along with the coordinates
img_path = "wb images/live_image_1724162934_white_balanced_image.jpg"
output_folder = "content\processed_images"
processed_image, coordinates = process_image(img_path, output_folder)

# Print the four coordinates
print("Four coordinates of the detected quadrilateral:", coordinates)
