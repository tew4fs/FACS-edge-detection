from PIL import Image

from algorithm import create_edge_detection_image

if __name__ == "__main__":
    image = Image.open("facs/assets/fennec.jpg")
    final_image = create_edge_detection_image(image)
    final_image.show()
