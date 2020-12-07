import sys
from datetime import datetime
from argparse import ArgumentParser

sys.path.insert(0, './')
from transfer_learning_model import *
from generators import *


def face_recognition_model(model_name: str, activation: str):
    # load the desired keras model with transfer learning enabled
    model = load_model(model_name, activation)

    training_data, validation_data = get_generator(model_name)

    history = model.fit(
        training_data,
        validation_data=validation_data,
        epochs=2,
        steps_per_epoch=len(training_data),
        validation_steps=len(validation_data)
    )

    save_time = datetime.now().strftime('%m-%d-%Y-%H_%M')
    model.save('../weights/{}_{}_{}'.format(model_name,
                                            activation,
                                            save_time))


if __name__ == '__main__':
    face_recognition_model('resnetv2', 'softmax')
