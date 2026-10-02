import cv2
import numpy as np


def preprocess(image, size=(224, 224), normalize=True):
    """
    Preprocess an image for a computer-vision pipeline.

    Steps:
    1. Resize
    2. Convert BGR to grayscale if necessary
    3. Normalize pixel values if requested

    Parameters:
        image: Input image as a NumPy array.
        size: Target image size as (width, height).
        normalize: Whether to normalize pixel values to 0-1.

    Returns:
        Processed image as a NumPy array.
    """

    # Step 1: Resize
    processed_image = cv2.resize(image, size)

    # Step 2: Convert to grayscale if image is BGR
    if len(processed_image.shape) == 3:
        processed_image = cv2.cvtColor(
            processed_image,
            cv2.COLOR_BGR2GRAY
        )

    # Step 3: Normalize if requested
    if normalize:
        processed_image = processed_image.astype(np.float32) / 255.0

    return processed_image