import os
import time
import cv2
import numpy as np
import pyrealsense2 as rs


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

if __name__ == "__main__":
    # Set the save path and filename
    save_path = "./images"  # Directory to save the image
    filename = "test_image"  # Filename for the saved image

    # Call the function to capture and save the cropped image
    try:
        capture_single_image(save_path=save_path, filename=filename)
    except Exception as e:
        print(f"An error occurred: {e}")
