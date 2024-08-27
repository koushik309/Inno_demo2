import cv2
import numpy as np

# Dictionary to store the coordinates for each camera ID
camera_coords = {
    0: np.array([[600, 0], [600, 2053], [3500, 2053], [3500, 0]], dtype=np.float32),
    1: np.array([[600, 0], [600, 2053], [2800, 2053], [2800, 0]], dtype=np.float32),
    2: np.array([[600, 0], [600, 2053], [3200, 2053], [3200, 0]], dtype=np.float32),
    3: np.array([[800, 0], [800, 2053], [3000, 2053], [3000, 0]], dtype=np.float32)
}

# Function to adjust and crop the image based on camera ID
def crop_image(image, camera_id):
    if camera_id not in camera_coords:
        print(f"Camera ID {camera_id} not found in coordinates dictionary.")
        return None
    
    # Get the coordinates for the specific camera ID
    pts = camera_coords[camera_id].copy()

    # Calculate the width of the bounding box
    width = np.linalg.norm(pts[3] - pts[0])

    # Increase the width by 10%
    extra_width = 0.1 * width

    # Adjust the x coordinates to expand the width
    pts[0][0] -= extra_width / 2  # Top-left x
    pts[1][0] -= extra_width / 2  # Bottom-left x
    pts[2][0] += extra_width / 2  # Bottom-right x
    pts[3][0] += extra_width / 2  # Top-right x

    # Ensure the new coordinates are within image boundaries
    pts[0][0] = max(pts[0][0], 0)
    pts[1][0] = max(pts[1][0], 0)
    pts[2][0] = min(pts[2][0], image.shape[1])
    pts[3][0] = min(pts[3][0], image.shape[1])

    # Get the bounding box of the adjusted quadrilateral
    x_min = int(min(pts[:, 0]))
    x_max = int(max(pts[:, 0]))
    y_min = int(min(pts[:, 1]))
    y_max = int(max(pts[:, 1]))

    # Crop the image using the new bounding box
    cropped_image = image[y_min:y_max, x_min:x_max]


    return cropped_image

