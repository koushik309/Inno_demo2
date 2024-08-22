import cv2
import numpy as np
import matplotlib.pyplot as plt

def adjust_white_balance(img, white_pixel_coords):
    """
    Adjust the white balance of an image based on a specific pixel that should be white.
    """
    # Extract the BGR values of the specified white pixel
    x, y = white_pixel_coords
    white_pixel_value = img[y, x, :]

    # Calculate the correction factor for each channel
    correction_factor = 255 / white_pixel_value

    # Apply the correction factor to each channel
    corrected_img = img.astype(np.float32)
    for c in range(3):  # Iteratiing over B, G, R channels
        corrected_img[:, :, c] *= correction_factor[c]

    # Clip values to be in the range [0, 255] and convert back to uint8
    corrected_img = np.clip(corrected_img, 0, 255).astype(np.uint8)

    return corrected_img

def show_pixel_position_and_adjust_white_balance(img_path, pixel_coords):
    """
    Display the image with a marker on the specified pixel and adjust the white balance.
    """
    # Load the image
    img = cv2.imread(img_path)
    # Convert from BGR to RGB for displaying with matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the original image with the pixel marked
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(img_rgb)
    ax[0].plot(pixel_coords[0], pixel_coords[1], 'ro', markersize=10)
    ax[0].set_title('Original Image with Target Pixel')

    # Adjust the white balance based on the specified pixel
    wb_img = adjust_white_balance(img, pixel_coords)
    wb_img_rgb = cv2.cvtColor(wb_img, cv2.COLOR_BGR2RGB)

    # Display the white-balanced image
    ax[1].imshow(wb_img_rgb)
    ax[1].set_title('White Balanced Image')

    plt.show()


img_path = '/content/live_image_1724162934.jpg'


pixel_coords = (700, 250)


show_pixel_position_and_adjust_white_balance(img_path, pixel_coords)