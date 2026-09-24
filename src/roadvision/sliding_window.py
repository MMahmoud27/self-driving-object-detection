"""Turning a classifier into a detector by brute force.

A classifier answers "what is this?"; detection also needs "where is it?". The
cheapest bridge is to slide a fixed-size window across the image, classify every
crop, and keep the ones that come back as a vehicle with high confidence.

It works, but it is slow (one forward pass per window), it only finds objects
that happen to match the window size, and the boxes it produces are snapped to
the step grid. Those limits are exactly what YOLO removes; see
:mod:`roadvision.yolo`.
"""

import numpy as np

from roadvision.data import normalize


def extract_windows(image, window_h=32, window_w=32, step_h=16, step_w=16):
    """Crop ``image`` into a grid of overlapping windows.

    Returns ``(windows, positions)`` where ``windows`` has shape
    ``(n, window_h, window_w, 3)`` and ``positions`` holds the matching
    ``(x, y)`` top-left corner of each window in the original image.

    A step smaller than the window size makes the windows overlap, so an object
    straddling a grid line still lands fully inside at least one crop.
    """
    windows, positions = [], []
    height, width = image.shape[0], image.shape[1]

    for y in range(0, height, step_h):
        for x in range(0, width, step_w):
            # Drop partial windows at the right and bottom edges: the classifier
            # takes a fixed input size and cannot handle a short crop.
            if y + window_h <= height and x + window_w <= width:
                windows.append(image[y:y + window_h, x:x + window_w])
                positions.append((x, y))

    return np.array(windows), positions


def classify_windows(model, windows):
    """Run ``model`` over every window.

    Returns ``(y_pred, confidence)``: the argmax class per window and the
    softmax probability the model assigned to it.
    """
    windows = np.array(windows)
    probabilities = model.predict(normalize(windows))
    return np.argmax(probabilities, axis=-1), np.max(probabilities, axis=-1)


def detect_vehicles(model, windows, positions=None, threshold=0.6):
    """Keep only the windows classified as a vehicle above ``threshold``.

    Class 0 is background, so ``y_pred > 0`` is the test for "some kind of
    vehicle". Returns a list of dicts with the window index, its class id and
    name, the confidence, and the ``(x, y)`` position when one was supplied.
    """
    from roadvision.data import IDX_TO_CLASS

    y_pred, confidence = classify_windows(model, windows)

    detections = []
    for i in range(len(y_pred)):
        if y_pred[i] > 0 and confidence[i] > threshold:
            detection = {
                "index": i,
                "label": int(y_pred[i]),
                "class_name": IDX_TO_CLASS[int(y_pred[i])],
                "confidence": float(confidence[i]),
            }
            if positions is not None:
                detection["x"], detection["y"] = positions[i]
            detections.append(detection)

    return detections


def window_accuracy(model, windows, labels):
    """Fraction of windows whose predicted class matches ``labels``.

    An earlier version of this helper called ``perceptron.predict`` directly
    instead of using its ``model`` argument, so passing a different model in
    silently re-scored the perceptron. That is fixed here, which is why the
    CNN's sliding-window number in ``docs/RESULTS.md`` is flagged as needing a
    re-run.
    """
    y_pred, _ = classify_windows(model, windows)
    return float(np.mean(y_pred == np.asarray(labels)))
