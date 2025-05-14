import cv2
import numpy as np
from skimage import measure
from typing import Tuple, List, Dict
import logging
from services.helper_scripts import preprocess_image
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SeedCounter:
    def __init__(self, min_seed_area: int = 500, max_seed_area: int = 10000):
        self.min_seed_area = min_seed_area  # Minimum area for a seed to be counted
        self.max_seed_area = max_seed_area  # Maximum area for a seed to be counted


    def detect_seeds(self, image: np.ndarray) -> Tuple[int, List[Dict]]:
        """
        Detect and count seeds in the image using connected components
        Returns the count and list of seed properties
        """
        # Find all connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)

        print(stats)
        print(num_labels)
        
        # Filter and count seeds
        seeds = []
        for i in range(1, num_labels):  # Start from 1 to skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if self.min_seed_area <= area <= self.max_seed_area:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                seeds.append({
                    'area': area,
                    'centroid': (centroids[i][1], centroids[i][0]),  # (y, x) format
                    'bbox': (y, x, y + h, x + w)  # (y1, x1, y2, x2) format
                })

        return len(seeds), seeds


    def process_image(self, image_path: str) -> Dict:
        """
        Process an image and return seed count and analysis
        """
        try:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image from {image_path}")

            # Preprocess the image to get binary mask
            binary_mask = preprocess_image(image, min_size=3000)

            cv2.imshow('binary_mask', binary_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Detect seeds using the binary mask
            count, seeds = self.detect_seeds(binary_mask)


            return {
                'total_seeds': count,
                'seeds': seeds,
                'status': 'success',
            }

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {
                'total_seeds': 0,
                'seeds': [],
                'status': 'error',
                'error': str(e)
            }

    def export_results(self, image: np.ndarray, seeds: List[Dict]) -> np.ndarray:
        """
        Create a visualization of the detected seeds
        """
        # Create a copy of the image for visualization
        vis_image = image.copy()

        # Draw bounding boxes around detected seeds
        for seed in seeds:
            y1, x1, y2, x2 = seed['bbox']
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(vis_image, 
                      (int(seed['centroid'][1]), int(seed['centroid'][0])), 
                      3, (0, 0, 255), -1)

        return vis_image 