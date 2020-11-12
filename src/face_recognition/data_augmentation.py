from tensorflow.keras.preprocessing.image import *
from tensorflow.keras.applications import *


def get_data_generator(augmentation: bool = False, train: bool = True,
                       model: str = None):

    if model == 'resnetv2':
        print('** resnetv2 augmentation **')
        if train:
            data_generator = ImageDataGenerator(
                dtype='float32',
                preprocessing_function=resnet_v2.preprocess_input,
                horizontal_flip=True,
                fill_mode="nearest",
                zoom_range=0.3,
                width_shift_range=0.1,
                height_shift_range=0.1,
                rotation_range=30)
        else:
            data_generator = ImageDataGenerator(
                dtype='float32',
                preprocessing_function=resnet_v2.preprocess_input,
                horizontal_flip=False,
                fill_mode="nearest")

        return data_generator

    # No Data augmentation
    if not augmentation:
        data_generator = ImageDataGenerator(
            rescale=1. / 255,
            horizontal_flip=False)

    # random horizontal flips, rotation and size changes
    elif augmentation:
        if train:
            data_generator = ImageDataGenerator(
                rescale=1. / 255,
                horizontal_flip=True,
                fill_mode="nearest",
                zoom_range=0.3,
                width_shift_range=0.1,
                height_shift_range=0.1,
                rotation_range=30)
        else:
            data_generator = ImageDataGenerator(
                rescale=1. / 255,
                horizontal_flip=False,
                fill_mode="nearest",
                zoom_range=0.,
                width_shift_range=0.,
                height_shift_range=0.,
                rotation_range=0.)

    return data_generator
