import cv2
import numpy as np
import sys
from argparse import ArgumentParser

sys.path.insert(0, './')
from facecrop import *


def capture_face(person):

    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    count = 0

    detector = FaceDetector("haarcascade_frontalface_default.xml", person)

    # Collect 100 samples of your face from webcam input
    while True:

        ret, frame = cap.read()
        face = detector.detect(frame, count)
        if face is not None:
            count += 1

            # Save file in specified directory with unique name
            file_name_path = './Images/' + str(count) + '.jpg'
            cv2.imwrite(file_name_path, face)

            # Put count on images and display live count
            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0), 2)
            cv2.imshow('Face Cropper', face)

        else:
            print("Face not found")
            pass

        if cv2.waitKey(1) == 13 or count == 100:  # 13 is the Enter Key
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Collecting Samples Complete")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-n", "--name",
                        help="What is you Name")

    args = parser.parse_args()
    person_name = args.name

    capture_face(person_name)
