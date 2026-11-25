"""Custom CNN architecture in TensorFlow/Keras for 128x128 Real vs Fake classification."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_model(
    input_shape: tuple[int, int, int] = (128, 128, 3), learning_rate: float = 1e-3
) -> tf.keras.Model:
    """Build and compile a custom multi-block CNN with BatchNorm, MaxPooling, and Dropout."""
    inputs = layers.Input(shape=input_shape, name="image_input")

    # Block 1: Low-level edge & pixel noise extraction
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Dropout(0.15, name="drop1")(x)

    # Block 2: Mid-level texture & boundary artifacts
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # Block 3: High-level semantic & synthesis artifact features
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = layers.Dropout(0.30, name="drop3")(x)

    # Block 4: Deep spatial aggregation
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4_1")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Dense classification head
    x = layers.Dense(128, activation="relu", name="fc1")(x)
    x = layers.BatchNormalization(name="bn_fc")(x)
    x = layers.Dropout(0.40, name="drop_fc")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="fake_probability")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="deepfake_detector_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model
