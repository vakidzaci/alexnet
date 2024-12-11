import cv2
import numpy as np

def segment_sentence_into_words(sentence_image):
    """
    Segments a sentence image into individual word images.

    Args:
        sentence_image (numpy.ndarray): Grayscale or binary image of the sentence.

    Returns:
        List[numpy.ndarray]: List of cropped word images.
    """
    # Ensure the image is binary
    if len(sentence_image.shape) == 3:  # If the image is not grayscale
        sentence_image = cv2.cvtColor(sentence_image, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(sentence_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Compute the vertical projection profile
    vertical_projection = np.sum(binary_image, axis=0)

    # Identify the boundaries where the projection is zero (valleys)
    threshold = 0  # Adjust if necessary for noisy images
    word_boundaries = []
    in_word = False
    start_idx = 0

    for i, value in enumerate(vertical_projection):
        if value > threshold and not in_word:
            start_idx = i
            in_word = True
        elif value <= threshold and in_word:
            word_boundaries.append((start_idx, i))
            in_word = False

    # Handle the last word if the image ends with text
    if in_word:
        word_boundaries.append((start_idx, len(vertical_projection)))

    # Crop word images based on identified boundaries
    word_images = []
    for start, end in word_boundaries:
        if end - start > 1:  # Filter out very small segments
            word_image = binary_image[:, start:end]
            word_images.append(word_image)

    return word_images
