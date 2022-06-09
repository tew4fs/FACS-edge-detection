from PIL import Image, ImageOps
im = Image.open("facs/assets/fennec.jpg")

grayscale = ImageOps.grayscale(im)
w, h = grayscale.size
intensity = [[0 for x in range(h)] for y in range(w)]
for i in range(w):
    for j in range(h):
        intensity[i][j] = grayscale.getpixel((i, j))
