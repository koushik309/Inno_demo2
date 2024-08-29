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

def get_status_and_recommendations(biomass: int, sick_spots: int, week: int) -> dict:
    if biomass is None or sick_spots is None or week is None:
        return {"status": "Error loading results", "recommendation": ""}
    
    # Determine the status based on the week and biomass
    if week == 1 and 0 <= biomass <= 10:
        status = "On Track"
    elif week == 2 and 10 <= biomass <= 25:
        status = "On Track"
    elif week == 3 and 25 <= biomass <= 40:
        status = "On Track"
    elif week == 4 and 40 <= biomass <= 80:
        status = "On Track"
    elif week == 5 and 80 <= biomass <= 100:
        status = "On Track"
    else:
        status = "Off Track"
    
    # Determine the recommendation based on the number of sick spots
    if 1<= sick_spots < 10:
        recommendation = "Abnormal plant pattern detected! Check individual plants for decontamination."
    elif sick_spots >= 10:
        recommendation = "Nutrition deficiency detected! Increase nutrition and water supply."
    else:
        recommendation = "No specific recommendation."

    return {'status': status, "recommendation": recommendation}
