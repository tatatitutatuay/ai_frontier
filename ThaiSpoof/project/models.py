from __future__ import annotations


def build_small_cnn(input_shape: tuple[int, int, int], learning_rate: float = 1e-4):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(96, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="thaispoof_small_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _residual_block(x, filters: int, stride: int = 1):
    from tensorflow.keras import layers

    shortcut = x
    y = layers.Conv2D(filters, (3, 3), strides=stride, padding="same", use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    return layers.Activation("relu")(layers.Add()([shortcut, y]))


def build_resnet_lite(input_shape: tuple[int, int, int], learning_rate: float = 1e-4):
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (5, 5), strides=2, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = _residual_block(x, 32)
    x = _residual_block(x, 64, stride=2)
    x = _residual_block(x, 64)
    x = _residual_block(x, 96, stride=2)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="thaispoof_resnet_lite")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(model_name: str, input_shape: tuple[int, int, int], learning_rate: float = 1e-4):
    model_name = model_name.lower()
    if model_name == "small_cnn":
        return build_small_cnn(input_shape, learning_rate)
    if model_name == "resnet_lite":
        return build_resnet_lite(input_shape, learning_rate)
    raise ValueError("model_name must be 'small_cnn' or 'resnet_lite'")
