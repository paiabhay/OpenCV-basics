import cv2
import matplotlib.pyplot as plt

# Read an image
img = cv2.imread('../Images/sunflower.jpg')

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def plot_image(input, is_gray=False):
    if not is_gray:
        img_gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        plt.imshow(img_gray)
    else:
        plt.imshow(input, cmap='gray')

    # Turn-off axis around image
    plt.axis('off')
    plt.show()

# Print pixel values
print('Pixel values of original image', img)
plot_image(gray_img, is_gray=True)

print('Pixel values of gray-scale image', gray_img)
plot_image(img, is_gray=False)

# Display a small patch of image
img_patch = gray_img[130:170, 130:170]
print('Pixel values of patch image', img)
plot_image(img_patch, is_gray=True)