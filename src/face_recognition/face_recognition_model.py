import sys
from datetime import datetime
from argparse import ArgumentParser

sys.path.insert(0, './')
from transfer_learning_model import *
from generators import *


def face_recognition_model(model_name: str):
    # load the desired keras model with transfer learning enabled
    model = load_model(model_name)

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy', 'mae']
    )

    training_data, validation_data = get_generator()

    history = model.fit(
        training_data,
        validation_data=validation_data,
        epochs=5,
        steps_per_epoch=len(training_data),
        validation_steps=len(validation_data)
    )

    save_time = datetime.now().strftime('%m-%d-%Y-%H_%M')
    model.save('../weights/{}_{}.h5'.format(model_name,
                                         save_time))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-n", "--name",
                        help="Which Model you want to train")

    args = parser.parse_args()
    model_name = args.name

    face_recognition_model(model_name)
