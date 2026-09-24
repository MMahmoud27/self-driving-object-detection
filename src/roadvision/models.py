"""The three classifier architectures the project compares.

They are deliberately a progression: a flat perceptron that throws away the
image's spatial layout, a small CNN that keeps it, and an ImageNet-pretrained
backbone that already knows what edges and wheels look like before it ever sees
CIFAR. Accuracy climbs at each step; see ``docs/RESULTS.md``.
"""

from tensorflow.keras import optimizers
from tensorflow.keras.applications import VGG16, VGG19, DenseNet121, ResNet50
from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential

IMAGE_SHAPE = (32, 32, 3)
NUM_CLASSES = 3

# The four ImageNet backbones the project's wrapper can pull in.
EXPERTS = {
    "VGG16": VGG16,
    "VGG19": VGG19,
    "ResNet50": ResNet50,
    "DenseNet121": DenseNet121,
}


def build_perceptron(input_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES,
                     learning_rate=1e-3, momentum=0.9):
    """Flatten -> Dense(128, relu) -> Dense(3, softmax).

    Flattening discards the fact that neighbouring pixels are related, which is
    why this model tops out well below the CNN.
    """
    model = Sequential(name="perceptron")
    model.add(Input(shape=input_shape))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
        metrics=["accuracy"],
    )
    return model


def build_cnn(input_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES,
              learning_rate=1e-3, momentum=0.95):
    """Conv2D(32, 3x3) -> ReLU -> MaxPool -> Dense(128) -> Dense(3, softmax)."""
    model = Sequential(name="cnn")
    model.add(Input(shape=input_shape))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
        metrics=["accuracy"],
    )
    return model


def build_transfer_classifier(name="VGG16", input_shape=IMAGE_SHAPE,
                              num_classes=NUM_CLASSES, trainable=True,
                              learning_rate=1e-3, momentum=0.9):
    """An ImageNet backbone with a fresh 3-class head bolted on.

    ``include_top=False`` drops the backbone's own 1000-way ImageNet classifier
    and keeps only the convolutional feature extractor, which is the part whose
    visual understanding transfers to a new task.
    """
    if name not in EXPERTS:
        raise ValueError(f"Unknown expert {name!r}; choose from {sorted(EXPERTS)}")

    expert_conv = EXPERTS[name](
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    for layer in expert_conv.layers:
        layer.trainable = trainable

    model = Sequential(name=f"transfer_{name.lower()}")
    model.add(expert_conv)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
        metrics=["accuracy"],
    )
    return model
