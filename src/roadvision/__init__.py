"""roadvision - object detection for self-driving cars.

The package factors out the code that the four project notebooks build up
incrementally, so the same functions can be reused from scripts and from the
Streamlit app instead of being pasted between notebooks.

Two independent pipelines live here:

* A **3-class vehicle classifier** (``background`` / ``car`` / ``truck``) trained
  on a subset of CIFAR-10, combined with a **sliding window** scan to turn
  classification into crude detection. See :mod:`roadvision.data`,
  :mod:`roadvision.models` and :mod:`roadvision.sliding_window`.
* **YOLOv3**, a pretrained single-shot detector covering the 80 COCO classes,
  applied to both stills and video. See :mod:`roadvision.yolo`.
"""

__version__ = "1.0.0"

from roadvision.data import (
    IDX_TO_CLASS,
    label_to_onehot,
    load_vehicle_dataset,
    normalize,
)

_LAZY = {
    "build_cnn": "roadvision.models",
    "build_perceptron": "roadvision.models",
    "build_transfer_classifier": "roadvision.models",
}


def __getattr__(name):
    """Import the Keras model builders only when one is actually asked for.

    Pulling in TensorFlow costs several seconds and a lot of memory, and the
    numpy-only halves of this package (normalisation, sliding windows, the YOLO
    post-processing maths) don't need it.
    """
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "IDX_TO_CLASS",
    "build_cnn",
    "build_perceptron",
    "build_transfer_classifier",
    "label_to_onehot",
    "load_vehicle_dataset",
    "normalize",
]
