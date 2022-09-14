from PIL import ImageOps
import numpy as np

def create_intensity_array(image):
    grayscaled_image = grayscale_image(image)
    intensity_array = get_pixel_intensity_array(grayscaled_image)
    return intensity_array

def grayscale_image(image):
    return ImageOps.grayscale(image)

def get_pixel_intensity_array(image):
    image_width, image_height = image.size
    intensity_array = np.empty([image_height, image_width], dtype=int)
    for row_index in range(image_height):
        for column_index in range(image_width):
            intensity_array[row_index][column_index] = image.getpixel((column_index, row_index))
    return intensity_array

