from tensorflow import keras
from glob import glob


def load_model(model_name: str):

    if 'vgg16' in model_name:
        vgg16 = keras.applications.VGG16(
            include_top=False,
            input_shape=(224, 224, 3),
            weights='imagenet'
        )

        for layer in vgg16.layers:
            layer.trainable = False

        number_classes = len(glob('../people/*'))

        x = keras.layers.Flatten()(vgg16.output)

        prediction = keras.layers.Dense(number_classes,
                                        activation='softmax')(x)

        model = keras.models.Model(inputs=vgg16.input,
                                   outputs=prediction)

        model.summary()

    return model

