import sys
from tensorflow.keras.preprocessing.image import *

sys.path.insert(0, './')
from augmentation import *


def get_generator():

    training_data_generator = get_data_generator(True)
    validation_data_generator = get_data_generator(True, False)

    training_directory = '../people/train'
    validation_directory = '../people/test'

    training_data = training_data_generator.flow_from_directory(
        training_directory,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    validation_data = validation_data_generator.flow_from_directory(
        validation_directory,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    return training_data, validation_data

