import matplotlib.pyplot as plt
from PIL import Image

def display_image(image_path):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def save_image(image, output_path):
    image.save(output_path)
    print(f"Image saved at {output_path}")
