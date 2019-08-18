# Source: https://bit.ly/2REzgnS
# Below code will help you split video into frames
# and store them for further processing

import cv2
import os

# Playing video from file:
video = cv2.VideoCapture('../Videos/video2.mp4')

# Create a directory to store the frames (images) from videos
try:
    if not os.path.exists('Video_to_Images'):
        os.makedirs('Video_to_Images')
except OSError:
    print('Error:\nCannot create directory with name Video_to_Images')


current_frame = 0
while True:
    # Capture frame-by-frame
    ret, frame = video.read()

    # If ret is False, means there is no more frame present.
    # Hence can break loop
    if not ret:
        break

    # Saves image of the current frame in jpg file
    name = './Video_to_Images/Frame' + str(current_frame) + '.jpg'
    print('Creating...' + name)
    cv2.imwrite(name, frame)

    # To stop overriding images
    current_frame += 1

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()