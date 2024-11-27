import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
from ImageProcessing import *
from Printer_Communications import *

def capture_single_image(save_path, filename="captured_image"):
    """
    Captures a single cropped image using Intel RealSense camera and saves it.
    :param save_path: Path to save the captured image
    :param filename: Filename for the saved image
    """
    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Configure color stream

    # Start the pipeline
    pipeline.start(config)

    try:
        print("Initializing camera. Waiting for focus...")
        time.sleep(2)  # Wait for 2 seconds to allow the camera to adjust and focus

        # Capture a single frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Could not get a color frame from the camera.")

        # Convert frame to a numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Crop the image using coordinates from the red rectangle
        crop_coords = (250, 100, 520, 350)  # x_start, y_start, x_end, y_end
        x_start, y_start, x_end, y_end = crop_coords
        cropped_image = color_image[y_start:y_end, x_start:x_end]

        # Ensure save path exists
        os.makedirs(save_path, exist_ok=True)

        # Save the cropped image
        filepath = os.path.join(save_path, f"{filename}.jpg")
        cv2.imwrite(filepath, cropped_image)
        print(f"Image saved to {filepath}")
        return filepath
    finally:
        # Stop the pipeline
        pipeline.stop()
        print("Camera pipeline stopped.")

# Example usage
if __name__ == "__main__":
    save_folder = "./images"
    z_height = 0  # Initialize z-height
    last_contours = None
    step_size = 5  # mm
    ip = "192.168.0.17"  # Replace with your printer's IP

    while True:
        # Get the current z-height from the printer (replace with actual G-code communication)
        current_z_height = request_z_position(ip)

        # Check if the printer has moved the specified step size
        if current_z_height - z_height >= step_size:
            z_height = current_z_height

            # Capture and save an image
            image_path = capture_single_image(save_folder, f"z_height_{z_height}")

            # Process the image
            edges = CannyProcess(image_path)
            contours = createContours(edges)

            # Compare contours if this is not the first image
            if last_contours is not None:
                similar = closestPointComparison(last_contours , contours, cutHeight = maximumHeight(last_contours) - 10)
                if not similar:
                    print(f"Contours differ at z-height {z_height}. Pausing printer.")
                    issue_gcode(ip, "M25")  # Pause the printer

            # Update last_contours
            last_contours = contours

        # Simulate exit condition (replace with actual logic or user input)
        if z_height > 50:  # Example: Stop after 50 mm
            break

        time.sleep(1)  # Delay between checks
