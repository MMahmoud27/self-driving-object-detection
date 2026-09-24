"""The 3-class vehicle dataset, carved out of CIFAR-10.

CIFAR-10 has ten classes; only two of them are vehicles we care about. We keep
``automobile`` (label 1) and ``truck`` (label 9), and fold the remaining eight
classes into a single ``background`` class so the classifier learns "not a
vehicle" as an explicit answer rather than being forced to guess.
"""

import numpy as np

IDX_TO_CLASS = ["background", "car", "truck"]

# CIFAR-10's own label indices for the two vehicle classes.
LABEL_CAR = 1
LABEL_TRUCK = 9

# Channel-wise mean and standard deviation of CIFAR, used to standardise inputs.
CIFAR_MEAN = np.array((0.4914, 0.4822, 0.4465))
CIFAR_STD = np.array((0.2023, 0.1994, 0.2010))


def load_cifar10():
    """Load raw CIFAR-10 with the labels squeezed to 1-D.

    Keras is imported here rather than at module scope so that ``normalize`` and
    ``label_to_onehot`` -- which are pure numpy -- can be used without paying for
    a TensorFlow import.
    """
    from tensorflow.keras.datasets import cifar10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    return (x_train, y_train.squeeze()), (x_test, y_test.squeeze())


def construct_vehicle_dataset(data, labels, images_per_class):
    """Rebuild a CIFAR-10 split as a balanced background/car/truck set.

    ``images_per_class`` images are taken from each of the three classes, so the
    result is balanced by construction and chance accuracy is exactly 1/3.
    """
    mask_car = labels == LABEL_CAR
    mask_truck = labels == LABEL_TRUCK
    mask_background = np.invert(mask_car | mask_truck)

    data_car = data[mask_car][:images_per_class]
    data_truck = data[mask_truck][:images_per_class]
    data_background = data[mask_background][:images_per_class]

    new_data = np.vstack((data_background, data_car, data_truck))
    new_labels = np.repeat(np.array([0, 1, 2]), images_per_class, axis=0)
    return new_data, new_labels


def load_vehicle_dataset(train_per_class=5000, test_per_class=1000):
    """Return ``(X_train, y_train), (X_test, y_test)`` for the 3-class problem.

    Defaults give 15,000 training and 3,000 test images of shape (32, 32, 3).
    """
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    train = construct_vehicle_dataset(x_train, y_train, train_per_class)
    test = construct_vehicle_dataset(x_test, y_test, test_per_class)
    return train, test


def normalize(data):
    """Scale pixels to [0, 1], then standardise per colour channel.

    Standardising makes every input feature occupy a similar range, which lets
    the optimiser use one learning rate for all of them and converge faster.
    """
    return (data / 255 - CIFAR_MEAN) / CIFAR_STD


def label_to_onehot(labels, num_classes=3):
    """Turn integer labels into one-hot rows, e.g. ``2 -> [0, 0, 1]``.

    ``categorical_crossentropy`` compares the model's softmax output against a
    probability vector, so the targets have to be in the same shape.
    """
    labels = np.asarray(labels).astype(int)
    onehot = np.zeros((len(labels), num_classes))
    onehot[np.arange(len(labels)), labels] = 1
    return onehot
