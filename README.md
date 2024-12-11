import cv2
import numpy as np

def segment_sentence_into_words(sentence_image, num_splits):
    """
    Segments a sentence image into a specified number of word images.

    Args:
        sentence_image (numpy.ndarray): Grayscale or binary image of the sentence.
        num_splits (int): The number of word segments to produce.

    Returns:
        List[numpy.ndarray]: List of cropped word images.
    """
    # Ensure the image is binary
    if len(sentence_image.shape) == 3:  # If the image is not grayscale
        sentence_image = cv2.cvtColor(sentence_image, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(sentence_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate the image to connect characters within words
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))  # Adjust kernel size as needed
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

    # Compute the vertical projection profile
    vertical_projection = np.sum(dilated_image, axis=0)

    # Identify non-zero regions in the projection
    non_zero_columns = np.where(vertical_projection > 0)[0]
    if len(non_zero_columns) == 0:
        return []  # No text found

    start_idx = non_zero_columns[0]
    end_idx = non_zero_columns[-1]

    # Split into equal segments based on the number of splits
    segment_width = (end_idx - start_idx) // num_splits
    word_images = []

    for i in range(num_splits):
        segment_start = start_idx + i * segment_width
        segment_end = start_idx + (i + 1) * segment_width if i < num_splits - 1 else end_idx
        word_image = binary_image[:, segment_start:segment_end]
        word_images.append(word_image)

    return word_images

# Example usage
if __name__ == "__main__":
    # Load a sentence image (replace 'sentence_image.png' with your file path)
    sentence_image = cv2.imread("sentence_image.png", cv2.IMREAD_GRAYSCALE)

    # Define the number of splits (words)
    num_words = 5  # Replace with the desired number of splits

    # Get word images
    words = segment_sentence_into_words(sentence_image, num_words)

    # Save and visualize the words
    for i, word in enumerate(words):
        cv2.imwrite(f"word_{i}.png", word)
        cv2.imshow(f"Word {i}", word)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
