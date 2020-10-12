import os
import sys
import cv2

sys.path.insert(0, '../face_utils')
from load_people import *


class FaceDetector(object):

    def __init__(self, xml_path):
        self.classifier = cv2.CascadeClassifier(xml_path)

    def detect(self, image):
        scale_factor = 1.2
        min_neighbors = 3
        min_size = (224, 224)
        faces_coord = self.classifier.detectMultiScale(image,
                                                       scaleFactor=scale_factor,
                                                       minNeighbors=min_neighbors,
                                                       minSize=min_size,
                                                       flags=cv2.CASCADE_SCALE_IMAGE)
        return faces_coord


def cut_faces(image, faces_coord):
    faces = []
    for (x, y, w, h) in faces_coord:
        faces.append(image[y: y + h, x: x + w])

    return faces


def resize(images, size=(224, 224)):
    images_norm = []
    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size,
                                    interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size,
                                    interpolation=cv2.INTER_CUBIC)
        images_norm.append(image_norm)

    return images_norm


def normalize_faces(image, faces_coord):
    faces = cut_faces(image, faces_coord)
    faces = resize(faces)

    return faces


people_dictionary = dataset()

for person in people_dictionary:
    directory_path = '../people/{}/crop'.format(person)
    if not os.path.exists(directory_path):
        os.mkdir('../people/{}/crop'.format(person))

    for index, image in enumerate(people_dictionary[person]):
        detector = FaceDetector("haarcascade_frontalface_default.xml")
        faces_coord = detector.detect(image)
        faces = normalize_faces(image, faces_coord)

        for i, face in enumerate(faces):
            file_name = '{}_{}_{}.jpeg'.format(person,
                                               index,
                                               i)
            file_path = os.path.join(directory_path, file_name)
            if not os.path.exists(file_path):
                cv2.imwrite(file_path, faces[i])
