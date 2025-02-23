import cv2 as cv
import numpy as np

def extract_hsv_ranges(image_path, region1_coords, region2_coords):
    """
    Extracts HSV color ranges for two regions in an image.

    :param image_path: Path to the input image.
    :param region1_coords: Coordinates of the first region (x, y, width, height) for Printer 1's filament.
    :param region2_coords: Coordinates of the second region (x, y, width, height) for Printer 2's filament.
    :return: Two tuples containing the lower and upper HSV bounds for each region.
    """
    # Load the image
    image = cv.imread(image_path)
    if image is None:
        raise ValueError("Image not found at the specified path.")

    # Convert to HSV color space
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Extract Region 1 (Printer 1's filament)
    x1, y1, w1, h1 = region1_coords
    region1 = hsv_image[y1:y1 + h1, x1:x1 + w1]
    lower_bound_printer1 = np.min(region1.reshape(-1, 3), axis=0)
    upper_bound_printer1 = np.max(region1.reshape(-1, 3), axis=0)

    # Extract Region 2 (Printer 2's filament)
    x2, y2, w2, h2 = region2_coords
    region2 = hsv_image[y2:y2 + h2, x2:x2 + w2]
    lower_bound_printer2 = np.min(region2.reshape(-1, 3), axis=0)
    upper_bound_printer2 = np.max(region2.reshape(-1, 3), axis=0)

    return (lower_bound_printer1.tolist(), upper_bound_printer1.tolist()), \
           (lower_bound_printer2.tolist(), upper_bound_printer2.tolist())


if __name__ == "__main__":
    # Path to the provided image
    image_path = "IMG_8725.jpg"  # Replace with your actual path if different

    # Define regions for each filament (adjust coordinates as necessary)
    # These coordinates are approximate and can be adjusted based on the specific areas of interest in your image.
    region_printer1 = (50, 50, 100, 100)  # Example: x=50, y=50, width=100, height=100
    region_printer2 = (200, 200, 100, 100)  # Example: x=200, y=200, width=100, height=100

    # Extract HSV ranges
    printer1_hsv_range, printer2_hsv_range = extract_hsv_ranges(image_path, region_printer1, region_printer2)

    print(f"Printer 1 HSV Range: {printer1_hsv_range}")
    print(f"Printer 2 HSV Range: {printer2_hsv_range}")
