from .transfer_learning_model import *
from .generators import *
from datetime import datetime


def face_recognition(model_name: str):

    # load the desired keras model with transfer learning enabled
    model = load_model(model_name)

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy', 'mae']
    )

    training_data, validation_data = get_generator()

    trained_model = model.fit(
        training_data,
        validation_data=validation_data,
        epochs=5,
        steps_per_epoch=len(training_data),
        validation_steps=len(validation_data)
    )

    save_time = datetime.now().strftime('%m-%d-%Y-%H:%M:')
    trained_model.save('../weights/{}_{}'.format(model_name,
                                                 save_time))
