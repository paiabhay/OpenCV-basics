import cv2

# Read an image
img = cv2.imread('../Images/sunflower.jpg')

# Convert RGB image to Gray Scale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Show Original Image
cv2.imshow('Image', img)
# Waitkey - Time delay of 2000 msec
cv2.waitKey(2000)
# Close the window
cv2.destroyAllWindows()

# Show Gray Scale image
cv2.imshow('Gray-Scale-Image', img_gray)
cv2.waitKey(2000)
cv2.destroyAllWindows()