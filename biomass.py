import cv2
import numpy as np
import os

def calculate_biomass(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity_green = 35
    lower_green = np.array([60 - sensitivity_green, 52, 72])
    upper_green = np.array([67 + sensitivity_green, 255, 255])

    mask_green = cv2.inRange(image_hsv, lower_green, upper_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

    total_pixels = image.shape[0] * image.shape[1]
    green_pixels = np.count_nonzero(mask)
    percentage_green = (green_pixels / total_pixels) * 100

    return percentage_green, mask

def save_green_extracted_image(image, mask, output_path):
    green_pixels = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite(output_path, green_pixels)
    print(f"Green extracted image saved to {output_path}")

def get_status_and_recommendation(biomass, sick_spots):
    status = "On Track" if biomass > 50 else "Off Track"
    if sick_spots < 10:
        recommendation = "Abnormal plant pattern detected -> Check individual plants for decontamination"
    else:
        recommendation = "Nutrition deficiency detected -> increase nutrition and water supply"
    return status, recommendation
