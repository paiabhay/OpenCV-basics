# This file will help to identify emotions in videos
'''
The video file is provided as input and
then we check each frame to identify the emotion
'''

# Helping libraries
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import os
import matplotlib as mpl
mpl.use('TkAgg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Control log message outputs


# Create a Neural Network
def create_model():
    # Create the model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    return model


def run_model(model_location, video_location, haar_cascade_xml):
    model = create_model()

    # Read model
    # We will be using already trained model
    model.load_weights(model_location)

    # Prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # Pictionary which assigns each label an emotion
    # Provide in alphabetical order
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                    3: "Happy", 4: "Neutral", 5: "Sad",
                    6: "Surprised"}

    # Provide input video
    # To read from webcam, use cv2.VideoCapture(0)
    video = cv2.VideoCapture(video_location)

    frame_counter = 0
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = video.read()

        # If no frame present then, break
        # ret - provides boolean value
        if not ret:
            break

        # Detecting face using haar_cascade_classifier
        face_cascade = cv2.CascadeClassifier(haar_cascade_xml)

        # Convert the frame to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # For faces detected, identify emotion
        for (x, y, w, h) in faces:
            # Draw rectangle box identifying the face in frame
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)

            # Convert the face to gray scale
            frame_img_gray = gray[y:y + h, x:x + w]

            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(frame_img_gray, (48, 48)), -1), 0)

            # Make a prediction
            predicted = model.predict(cropped_img)
            max_index = int(np.argmax(predicted))

            # Print predicted emotion (the one with maximum likelihood
            print('Frame {}, Emotion: {}'.format(frame_counter, emotion_dict[max_index]))
            frame_counter +=1

            # Display the emotion text in video by marking the face
            cv2.putText(frame, emotion_dict[max_index], (x + 20, y - 60),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255),
                        2, cv2.LINE_AA)

        cv2.imshow('Video', frame)

        # If you want to manually end video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


def main():
    # Provide locations of files used for in below code
    video_location = 'Videos/video3.mp4'
    model_location = 'Saved_Models/model.h5'
    haar_cascade_xml = 'Haarcascades/haarcascade_frontalface_default.xml'

    # Run the model
    run_model(model_location, video_location, haar_cascade_xml)


if __name__ == '__main__':
    main()