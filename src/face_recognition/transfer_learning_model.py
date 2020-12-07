from tensorflow import keras
from glob import glob


def load_model(model_name: str, activation: str):
    number_classes = len(glob('../people/train/*'))
    if model_name == 'vgg16':
        vgg16 = keras.applications.VGG16(
            include_top=False,
            input_shape=(224, 224, 3),
            weights='imagenet'
        )

        for layer in vgg16.layers:
            layer.trainable = False

        x = keras.layers.Flatten()(vgg16.output)

        prediction = keras.layers.Dense(number_classes,
                                        activation=activation)(x)
        # sigmoid?

        model = keras.models.Model(inputs=vgg16.input,
                                   outputs=prediction)

    if model_name == 'resnetv2':

        restnetv2 = keras.applications.ResNet50V2(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3))
        output = restnetv2.layers[-1].output
        output = keras.layers.Flatten()(output)
        restnetv2 = keras.models.Model(inputs=restnetv2.input, outputs=output)
        for layer in restnetv2.layers:
            layer.trainable = False
        restnetv2.summary()
        model = keras.models.Sequential()
        model.add(restnetv2)
        model.add(keras.layers.Dense(512, activation='relu',
                                     input_dim=(224, 224, 3)))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(number_classes, activation=activation))

    sgd = keras.optimizers.SGD(
            learning_rate=0.001,
            momentum=0.9)

    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D) or \
                isinstance(layer, keras.layers.Dense):
            layer.kernel_regularizer = keras.regularizers.l2(0.001)

    model.compile(
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer=sgd,
        metrics=['accuracy', 'mae']
    )

    model.summary()

    return model
