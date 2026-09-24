"""Plots used to judge the classifiers: sample images, curves, confusion."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from roadvision.data import IDX_TO_CLASS


def plot_one_image(data, labels=None, index=None, image_shape=(32, 32, 3),
                   fig_size=None):
    """Show a single image, from either one array or a stack plus an index."""
    data = np.asarray(data)

    if data.ndim == 1:
        data = data.reshape(image_shape)
    elif data.ndim == 2:
        data = data.reshape([-1, *image_shape])

    if data.ndim == 3:
        image = data
        label = labels if labels is not None and not hasattr(labels, "__len__") else ""
        if hasattr(labels, "__len__") and len(labels) == 1:
            label = labels[0]
    elif data.ndim == 4:
        if index is None:
            raise ValueError("For a stack of images, supply an 'index'.")
        image = data[index]
        # Distinguish a per-image label array from the 3-item class-name map.
        if labels is not None and hasattr(labels, "__len__") and len(labels) == data.shape[0]:
            label = labels[index]
        else:
            label = ""
    else:
        raise ValueError(f"Expected 3 or 4 dimensions, got shape {data.shape}")

    if fig_size is not None:
        plt.figure(figsize=fig_size)
    plt.title(f"Label: {label}" if label != "" else "")
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def plot_acc(history, ax=None, xlabel="Epoch #", chance=1 / 3):
    """Training vs validation accuracy per epoch.

    The gap between the two curves is the overfitting story: when training keeps
    climbing while validation flattens, the model is memorising rather than
    generalising. The chance line marks what guessing would score.
    """
    history = history.history if hasattr(history, "history") else history
    history = dict(history)
    history["epoch"] = list(range(len(history["val_accuracy"])))
    frame = pd.DataFrame.from_dict(history)

    best_epoch = frame.sort_values(by="val_accuracy", ascending=False).iloc[0]["epoch"]

    if ax is None:
        _, ax = plt.subplots(1, 1)
    sns.lineplot(x="epoch", y="val_accuracy", data=frame, label="Validation", ax=ax)
    sns.lineplot(x="epoch", y="accuracy", data=frame, label="Training", ax=ax)
    ax.axhline(chance, linestyle="--", color="red", label="Chance")
    ax.axvline(x=best_epoch, linestyle="--", color="green", label="Best Epoch")
    ax.legend(loc=1)
    ax.set_ylim([0.01, 1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy (Fraction)")
    plt.show()
    return ax


def plot_confusion_matrix(y_true, y_pred, labels=None):
    """Heatmap of true vs predicted class.

    Overall accuracy hides *which* classes a model confuses; here the car/truck
    off-diagonal is the interesting cell, since both are boxy vehicles at 32x32.
    """
    labels = labels or [name.capitalize() for name in IDX_TO_CLASS]
    matrix = confusion_matrix(y_true, y_pred)

    frame = pd.DataFrame(matrix, index=labels, columns=labels)
    plt.figure(figsize=(7, 6))
    sns.heatmap(frame, annot=True, cmap="YlGnBu", fmt="g")
    plt.yticks(np.arange(len(labels)) + 0.5, labels, va="center")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()
    return matrix
