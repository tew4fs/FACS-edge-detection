from PIL import Image
import numpy as np

from facs.image_processor import create_intensity_array
from config.constants import NUMBER_OF_ITERATIONS


def create_edge_detection_image(image):
    intensity_array = create_intensity_array(image)
    print(NUMBER_OF_ITERATIONS)
    # heuristics_array = get_heuristics_array(intensity_array)
    final_image = Image.fromarray((intensity_array).astype(np.uint8))
    return final_image


def get_heuristics_array(intensity_array):
    image_height = intensity_array.shape[0]
    image_width = intensity_array.shape[1]
    variation_array = get_variation_array(intensity_array)
    max_variation = get_max_variation(variation_array)
    heuristics_array = np.empty([image_height, image_width])
    for row_index in range(image_height):
        for column_index in range(image_width):
            heuristics_array[row_index][column_index] = variation_array[row_index][column_index] / max_variation
    return heuristics_array


def get_variation_array(intensity_array):
    image_height = intensity_array.shape[0]
    image_width = intensity_array.shape[1]
    variation_array = np.empty([image_height, image_width], dtype=int)
    for row_index in range(image_height):
        for column_index in range(image_width):
            variation_array[row_index][column_index] = _get_variation_value(intensity_array, row_index, column_index)
    return variation_array


def get_max_variation(variation_array):
    max_variation = 0
    for column in variation_array:
        for value in column:
            if value > max_variation:
                max_variation = value
    return max_variation


def _get_variation_value(intensity_array, row_index, column_index):
    variation_value = (
        abs(
            _get_intensity_value(intensity_array, row_index - 1, column_index - 1)
            - _get_intensity_value(intensity_array, row_index + 1, column_index + 1)
        )
        + abs(
            _get_intensity_value(intensity_array, row_index, column_index - 1)
            - _get_intensity_value(intensity_array, row_index, column_index + 1)
        )
        + abs(
            _get_intensity_value(intensity_array, row_index + 1, column_index - 1)
            - _get_intensity_value(intensity_array, row_index - 1, column_index + 1)
        )
        + abs(
            _get_intensity_value(intensity_array, row_index + 1, column_index)
            - _get_intensity_value(intensity_array, row_index - 1, column_index)
        )
    )
    return variation_value


def _get_intensity_value(intensity_array, row_index, column_index):
    image_height = intensity_array.shape[0]
    image_width = intensity_array.shape[1]
    if row_index < 0 or column_index < 0 or row_index >= image_height or column_index >= image_width:
        return 0
    intensity_value = intensity_array[row_index][column_index]
    return intensity_value
