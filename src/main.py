from PIL import Image

from facs.algorithm import create_edge_detection_image

if __name__ == "__main__":
    image = Image.open("src/facs/assets/watermelon.png")
    final_image = create_edge_detection_image(image)
    final_image.show()
