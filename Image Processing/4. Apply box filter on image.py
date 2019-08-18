import cv2
import matplotlib.pyplot as plt


def apply_box_filter(image, kernal_size):
    # Kernal based box-blurring
    blur_image = cv2.blur(image, kernal_size)

    '''
    Gaussian blur is a a non-linear filter that enhances the effect of center pixel
    and gradually reduces the effect as the pixel gets farther from center.
    '''
    gaussian_blur_image = cv2.GaussianBlur(image, kernal_size, sigmaX=2)
    return blur_image, gaussian_blur_image


def plot_images(input_image, blur_image, gaussian_blur_image):
    # Converts an image from BGR to RGB and plot
    fig, ax = plt.subplots(nrows=1, ncols=3)

    ax[0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Input Image')
    ax[0].axis('off')

    ax[1].imshow(cv2.cvtColor(blur_image, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Box-Filtered Image (15x15)')
    ax[1].axis('off')

    ax[2].imshow(cv2.cvtColor(gaussian_blur_image, cv2.COLOR_BGR2RGB))
    ax[2].set_title('Gaussian Image (15x15)')
    ax[2].axis('off')
    plt.show()


def main():
    # Read an image
    img = cv2.imread('../Images/sunflower_1.jpg')
    # Initializing kernal Size - should always be odd x odd number
    kernal_size = (15, 15)

    blur_img, gaussian_blur_img = apply_box_filter(img, kernal_size)

    # Display images
    plot_images(img, blur_img, gaussian_blur_img)


if __name__ == '__main__':
    main()