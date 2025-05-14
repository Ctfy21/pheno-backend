import cv2
import numpy as np
from services.helper_scripts import remove_small_objects
import os
from os.path import isfile, join
from os import listdir
from services.helper_scripts import preprocess_image

def _crop_uid_image(binary_image: np.ndarray, original_image: np.ndarray) -> np.ndarray:
    """
    Crop the image into quarters and remove black parts
    """
    height, width = original_image.shape[:2]
    # Calculate the center point
    center_y, center_x = height // 3, width // 3
    
    # Crop the image to get the top-left quarter
    cropped = binary_image[:center_y, :center_x]

    cropped = remove_small_objects(cropped, min_size=600)

    # Create empty canvas
    canvas = np.zeros((height, width), dtype=np.uint8)
    
    # Place cropped image in the top-left corner of canvas
    canvas[:center_y, :center_x] = cropped
    
    canvas = remove_small_objects(canvas, min_size=10000)

    # Convert canvas to 3 channels
    canvas_3channel = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    
    # Apply mask to original image
    final_image = cv2.bitwise_and(canvas_3channel, original_image)
    
    # Find non-zero points in the binary mask
    non_zero = cv2.findNonZero(canvas)
    if non_zero is not None:
        # Get the bounding rectangle of non-zero points
        x, y, w, h = cv2.boundingRect(non_zero)
        # Crop the image to the bounding rectangle
        final_image = final_image[y:y+h, x:x+w]

    # Convert final_image to grayscale
    gray = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding
    binary_final = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    binary_final = remove_small_objects(cv2.bitwise_not(binary_final), min_size=35)

    return binary_final
        
def _crop_uid(image_path):
    
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    if not os.path.exists('results/uid_crops'):
        os.makedirs('results/uid_crops')
        
    # Process an image
    if not os.path.exists(image_path):
        print(f"Please place a test image at {image_path}")
        return
    
    image = cv2.imread(image_path)
    # Save visualization with proper path handling
    output_filename = os.path.basename(image_path)
    output_path_uid_crop = os.path.join('results', 'uid_crops', f'{os.path.splitext(output_filename)[0]}_uid_crop.jpg')
    
    cv2.imwrite(output_path_uid_crop, _crop_uid_image(preprocess_image(image), image))
    print(f"UID crop saved to {output_path_uid_crop}")

def get_uids(path_to_images: str):
    onlyfiles = [f for f in listdir(path_to_images) if isfile(join(path_to_images, f))]
    for file in onlyfiles:
        image_path = join(path_to_images, file)
        _crop_uid(image_path)

    