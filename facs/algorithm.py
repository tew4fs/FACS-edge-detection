from PIL import Image

from image_processor import create_intensity_array

def create_edge_detection_image(image):
    intensity_array = create_intensity_array(image)
    final_image = Image.fromarray(intensity_array)
    return final_image
