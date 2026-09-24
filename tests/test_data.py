"""Carving the 3-class vehicle set out of CIFAR-10.

Driven by a synthetic stand-in for CIFAR rather than the real download, so this
runs offline and in CI.
"""

import numpy as np
import pytest

from roadvision.data import (
    IDX_TO_CLASS,
    LABEL_CAR,
    LABEL_TRUCK,
    construct_vehicle_dataset,
    label_to_onehot,
)


@pytest.fixture
def fake_cifar():
    """500 images across CIFAR-10's ten classes, 50 of each.

    Every image is filled with its own label value, so it is possible to tell
    afterwards which class each row came from.
    """
    labels = np.repeat(np.arange(10), 50)
    data = np.zeros((500, 32, 32, 3), dtype=np.uint8)
    for i, label in enumerate(labels):
        data[i] = label
    return data, labels


class TestVehicleDataset:
    def test_three_balanced_classes(self, fake_cifar):
        data, labels = fake_cifar
        x, y = construct_vehicle_dataset(data, labels, images_per_class=50)

        assert x.shape == (150, 32, 32, 3)
        assert y.shape == (150,)
        assert np.bincount(y).tolist() == [50, 50, 50]

    def test_class_names_line_up_with_indices(self):
        assert IDX_TO_CLASS == ["background", "car", "truck"]

    def test_cars_come_from_the_automobile_class(self, fake_cifar):
        data, labels = fake_cifar
        x, y = construct_vehicle_dataset(data, labels, images_per_class=50)
        # Images are filled with their original CIFAR label.
        assert (x[y == 1] == LABEL_CAR).all()

    def test_trucks_come_from_the_truck_class(self, fake_cifar):
        data, labels = fake_cifar
        x, y = construct_vehicle_dataset(data, labels, images_per_class=50)
        assert (x[y == 2] == LABEL_TRUCK).all()

    def test_background_excludes_both_vehicle_classes(self, fake_cifar):
        data, labels = fake_cifar
        x, y = construct_vehicle_dataset(data, labels, images_per_class=50)
        background = x[y == 0]
        assert not (background == LABEL_CAR).any()
        assert not (background == LABEL_TRUCK).any()

    def test_requesting_fewer_images_per_class(self, fake_cifar):
        data, labels = fake_cifar
        x, y = construct_vehicle_dataset(data, labels, images_per_class=10)
        assert x.shape[0] == 30
        assert np.bincount(y).tolist() == [10, 10, 10]

    def test_chance_accuracy_is_one_third(self, fake_cifar):
        # The set is balanced by construction, which is what makes the 33.3%
        # baseline in the README meaningful.
        data, labels = fake_cifar
        _, y = construct_vehicle_dataset(data, labels, images_per_class=50)
        counts = np.bincount(y)
        assert (counts == counts[0]).all()


class TestOneHot:
    def test_encodes_each_class(self):
        assert label_to_onehot([0, 1, 2]).tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def test_rows_sum_to_one(self):
        onehot = label_to_onehot([0, 1, 2, 2, 1, 0])
        assert (onehot.sum(axis=1) == 1).all()

    def test_accepts_a_numpy_array(self):
        assert label_to_onehot(np.array([2, 0])).tolist() == [[0, 0, 1], [1, 0, 0]]

    def test_round_trips_through_argmax(self):
        labels = [0, 1, 2, 1, 0, 2]
        assert label_to_onehot(labels).argmax(axis=1).tolist() == labels
