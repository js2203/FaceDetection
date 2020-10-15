import os
from glob import glob
from tensorflow import keras
from PIL import Image
import numpy as np
import cv2
import sys
from argparse import ArgumentParser

sys.path.insert(0, '../face_utils')
from facecrop import *


def face_recognition(model_name: str):

    model = keras.models.load_model('../weights/'+model_name)

    classifier = cv2.CascadeClassifier(
        '../face_utils/haarcascade_frontalface_default.xml')

    classes = []
    for name in glob('../people/train/*'):
        classes.append(os.path.basename(name))
    label = sorted(classes)

    # Initialize Webcam
    video_capture = cv2.VideoCapture(0)

    while True:

        _, frame = video_capture.read()

        faces = classifier.detectMultiScale(
            frame,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE)

        if faces is ():
            pass
        else:
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255),2)
            faces_only = normalize_faces(frame, faces)
            for face in faces_only:
                image = Image.fromarray(face, 'RGB')
                image_array = np.array(image)
                image_array = np.expand_dims(image_array, axis=0)
                prediction = model.predict(image_array)

            print(np.argmax(prediction))
            predicted_name = label[np.argmax(prediction)]

            cv2.putText(frame, predicted_name, (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0), 2)

        cv2.imshow('Detection', frame)
        if cv2.waitKey(1) == 13:  # 13 is the Enter Key
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    face_recognition('vgg16_10-15-2020-12_13.h5')
