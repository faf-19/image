from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    image = Image.open(image_path)  
    if image is None:
        print("Error: Could not load the image. Please check the file path.")
    return image

def separate_color_channels(image):
    r, g, b = image.split()  

    r = np.array(r)
    g = np.array(g)
    b = np.array(b)

    plt.figure(figsize=(18, 4))
    
    plt.subplot(1, 3, 1)
    plt.title('Red Channel')
    plt.imshow(np.stack([r, np.zeros_like(g), np.zeros_like(b)], axis=-1))  # Red channel
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Green Channel')
    plt.imshow(np.stack([np.zeros_like(r), g, np.zeros_like(b)], axis=-1))  # Green channel
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Blue Channel')
    plt.imshow(np.stack([np.zeros_like(r), np.zeros_like(g), b], axis=-1))  # Blue channel
    plt.axis('off')

    plt.savefig('channels_separated_pillow.png')
    plt.show()

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title('Red Image')
    plt.imshow(np.stack([r, np.zeros_like(g), np.zeros_like(b)], axis=-1))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Green Image')
    plt.imshow(np.stack([np.zeros_like(r), g, np.zeros_like(b)], axis=-1))
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Blue Image')
    plt.imshow(np.stack([np.zeros_like(r), np.zeros_like(g), b], axis=-1))
    plt.axis('off')

    plt.savefig('color_images_pillow.png')
    plt.show()

def convert_to_grayscale(image):
    gray_image = image.convert("L")  
    gray_image.save('gray_image_pillow.png')
    return gray_image

def create_negative_image(gray_image):
    negative_image = Image.eval(gray_image, lambda x: 255 - x)  
    negative_image.save('negative_image_pillow.png')

def main():
    image_path = 'photo.jpg'  
    image = load_image(image_path)
    if image is not None:
        separate_color_channels(image)
        gray_image = convert_to_grayscale(image)
        create_negative_image(gray_image)

        print("Images processed and saved: channels_separated_pillow.png, gray_image_pillow.png, negative_image_pillow.png, color_images_pillow.png.")

if __name__ == "__main__":
    main()