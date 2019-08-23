# This file will help to identify emotions in videos
'''
The video file is provided as input and
then we check each frame to identify the emotion
'''

# Helping libraries
import pandas as pd
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import spline
mpl.use('TkAgg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Control log message outputs

# Arrays to store frame and emotion
frame_number = list()
emotions_each_frame = list()
emotion_value = list()


def value_for_emotion(emotion):
    if emotion == "Angry":
        value = -2
    elif emotion == "Disgusted":
        value = -4
    elif emotion == "Fearful":
        value = -1
    elif emotion == "Happy":
        value = 3
    elif emotion == "Neutral":
        value = 0
    elif emotion == "Sad":
        value = -3
    else: # for 'Surprised'
        value = 2

    return value


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

    # Dictionary which assigns each label an emotion
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
            emo_val = value_for_emotion(emotion_dict[max_index])

            # Print predicted emotion (the one with maximum likelihood
            print('Frame {}, Emotion: {}, Value: {}'.format(frame_counter, emotion_dict[max_index], emo_val))
            frame_number.append(frame_counter)
            emotions_each_frame.append(emotion_dict[max_index])
            emotion_value.append(emo_val)

            frame_counter += 1

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
    return frame_number, emotions_each_frame, emotion_value


def save_to_csv(frames, emotions, emotion_val, video_file_name):
    # Create synthetic data folder if it doesn't exist
    if not os.path.exists('Extracted_Data'):
        os.makedirs('Extracted_Data')

    # Convert data to data-frame and save to csv file
    dataframe_emotions = pd.DataFrame({'Frames': frames,
                                       'Emotion': emotions,
                                       'Emotion_Value': emotion_val})
    dataframe_emotions.to_csv('Extracted_Data/{}_{}.csv'.format(video_file_name,
                                                                datetime.now().strftime('%d.%m.%Y_%H.%M.%S')),
                              index=False, header=True,
                              sep=',')
    print('Data saved to file successfully.')
    return dataframe_emotions


def plot_graph(dataframe_emotions):
    plt.figure(figsize=(10,20))
    plt.xlabel('Frames')
    plt.ylabel('Emotion_Value')
    plt.axhline(y=0, color='red')

    x = dataframe_emotions['Frames']
    y = dataframe_emotions['Emotion_Value']

    x_new = np.linspace(x.min(), x.max(), 500)
    y_smooth = spline(x, y, x_new)

    plt.plot(x_new, y_smooth, '-', color='lightcoral')
    plt.scatter(x, y, color='maroon')
    plt.show()

    # Plot Value counts
    plt.xlabel('Emotions')
    plt.ylabel('Frequency')
    dataframe_emotions['Emotion'].value_counts().plot()
    plt.show()


def main():
    # Provide locations of files used for in below code
    video_name = 'video3'
    video_location = 'Videos/video3.mp4'
    model_location = 'Saved_Models/model.h5'
    haar_cascade_xml = 'Haarcascades/haarcascade_frontalface_default.xml'

    # Run the model
    frames, emotions, emotion_val = run_model(model_location, video_location, haar_cascade_xml)

    df_emotions = save_to_csv(frames, emotions, emotion_val, video_name)
    print('Frequency count of each emotion: \n', df_emotions['Emotion'].value_counts())

    plot_graph(df_emotions)


if __name__ == '__main__':
    main()