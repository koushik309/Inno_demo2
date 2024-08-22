# camera.py
import cv2
import os
from pathlib import Path

def capture_image(output_path, camera_id=0):
    # Open a connection to the specified camera
    cap = cv2.VideoCapture(camera_id)  # Use the provided camera_id

    # Set the resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}.")
        return None

    # Capture a single frame
    ret, frame = cap.read()

    if ret:
        # Save the captured image to the specified path
        cv2.imwrite(output_path, frame)
        print(f"Image saved to {output_path}")
    else:
        print("Error: Could not capture image.")
        return None

    # Release the camera
    cap.release()
    return output_path