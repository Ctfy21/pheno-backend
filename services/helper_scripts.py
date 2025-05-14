import cv2
import numpy as np

def close_contours(image: np.ndarray) -> np.ndarray:
    """
    Close contours in the binary image using Convex Hull
    """
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a blank image
    closed = np.zeros_like(image)
    
    # Process each contour
    for contour in contours:
        # Calculate convex hull
        hull = cv2.convexHull(contour)
        # Draw filled convex hull
        cv2.drawContours(closed, [hull], -1, 255, -1)

    return closed

def remove_small_objects(image: np.ndarray, min_size: int = 50) -> np.ndarray:
    """
    Remove small connected objects from binary image
    """
    # Find all connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)
    
    # Create output image
    output = np.zeros_like(image)
    
    # Keep only components larger than min_size
    for i in range(1, num_labels):  # Start from 1 to skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            output[labels == i] = 255
            
    return output

def preprocess_image(image: np.ndarray, min_size: int = 50) -> np.ndarray:
    """
    Preprocess the image for seed detection
    """
    # Convert to grayscale if the image is in color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Remove small objects
    cleaned = remove_small_objects(thresh, min_size=70)

    # Perform morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # Close contours
    closed = close_contours(cleaned)

    # Remove small objects
    cleaned = remove_small_objects(closed, min_size=min_size)

    return cleaned 