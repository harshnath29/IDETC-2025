import os
from ImageProcessing import CannyProcess, createContours
import cv2 as cv
import time
from Printer_Communications import request_z_position
from app import capture_single_image

def detect_boundary_gap_by_color(save_path, ip1, ip2, step_size=2, distance_threshold=15):
    """
    Detects gaps in the boundary between two cooperative 3D prints using color-based edge detection.

    :param save_path: Path to save captured images
    :param ip1: IP address of the first printer
    :param ip2: IP address of the second printer
    :param step_size: Step size (in mm) for capturing images
    :param distance_threshold: Threshold for detecting gaps between edges (in pixels)
    :return: Boolean (True if boundary is good, False if a gap is detected)
    """
    # Initialize variables
    z_height1 = 0  # Initial Z height for printer 1
    z_height2 = 0  # Initial Z height for printer 2

    while True:
        # Get current Z heights from both printers
        current_z_height1 = request_z_position(ip1)
        current_z_height2 = request_z_position(ip2)

        # Check if printers have moved the specified step size
        if (current_z_height1 - z_height1 >= step_size) and (current_z_height2 - z_height2 >= step_size):
            z_height1 = current_z_height1
            z_height2 = current_z_height2

            # Capture images from both printers
            image_path1 = capture_single_image(save_path, f"printer1_z_{z_height1}")
            image_path2 = capture_single_image(save_path, f"printer2_z_{z_height2}")

            # Define color ranges for each filament (adjust HSV ranges as needed)
            color_range_printer1 = ([100, 50, 50], [140, 255, 255])  # Example: Blue filament
            color_range_printer2 = ([0, 50, 50], [10, 255, 255])     # Example: Red filament

            # Process images to extract edges for each printer's filament color
            edges_printer1 = CannyProcess(image_path1, color_range_printer1)
            edges_printer2 = CannyProcess(image_path2, color_range_printer2)

            # Create contours from edges
            contours_printer1 = createContours(edges_printer1)
            contours_printer2 = createContours(edges_printer2)

            # Compare last side contour of printer 1 with first side contour of printer 2
            if contours_printer1 and contours_printer2:
                last_contour_printer1 = contours_printer1[-1]
                first_contour_printer2 = contours_printer2[0]

                gap_detected = compare_contours(last_contour_printer1, first_contour_printer2, distance_threshold)

                if gap_detected:
                    print(f"Gap detected at Z height {z_height1}.")
                    return False

        # Simulate exit condition (e.g., stop after reaching a certain Z height)
        if z_height1 > 50 or z_height2 > 50:  # Example: Stop after 50 mm
            break

        time.sleep(0.5)  # Delay between checks

    print("No gaps detected in boundary.")
    return True


def compare_contours(contour1, contour2, distance_threshold):
    """
    Compares two contours to detect gaps based on a distance threshold.

    :param contour1: Contour from the first print (last side)
    :param contour2: Contour from the second print (first side)
    :param distance_threshold: Maximum allowed distance between points (in pixels)
    :return: Boolean (True if a gap is detected, False otherwise)
    """
    for point in contour1:
        distances = [cv.norm(point[0] - p[0]) for p in contour2]
        if min(distances) > distance_threshold:
            return True
    return False


if __name__ == "__main__":
    # Example usage of the function
    save_folder = "./images"
    printer_ip1 = "192.168.0.187"  # Replace with actual IP of Printer 1
    printer_ip2 = "192.168.0.188"  # Replace with actual IP of Printer 2

    result = detect_boundary_gap_by_color(save_folder, printer_ip1, printer_ip2)
    print(f"Boundary check result: {result}")
