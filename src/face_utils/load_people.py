import cv2
import os


def dataset():
    people_dictionary = {}
    people = [person for person in os.listdir("../raw_people/")]
    for i, person in enumerate(people):
        people_dictionary[person] = []
        for image in os.listdir("../raw_people/" + person):
            people_dictionary[person].append(
                cv2.imread("../raw_people/" + person + '/' + image, 0))

    return people_dictionary
