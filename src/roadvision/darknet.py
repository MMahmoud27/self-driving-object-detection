"""The YOLOv3 network, built in Keras and loaded from official Darknet weights.

YOLOv3 ships as a ``.weights`` file: a short header followed by a flat stream of
float32 values, with no architecture and no layer names. To use it from Keras
the graph has to be rebuilt exactly as Darknet defines it, and the stream read
back in the same order the layers were written.

That is what this module does, so the project depends only on the canonical
weights release rather than on a prebuilt ``.h5`` from anywhere else.

    from roadvision.darknet import build_yolov3, load_darknet_weights

    model = build_yolov3()
    load_darknet_weights(model, "data/yolov3.weights")
    model.save("data/yolo_weights.h5")

``scripts/download_assets.py`` does exactly this and caches the result.
"""

import numpy as np
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Input,
    LeakyReLU,
    UpSampling2D,
    ZeroPadding2D,
)
from tensorflow.keras.models import Model

# Darknet numbers every convolution in one continuous sequence, and the weights
# stream follows that order. The three detection heads (81, 93, 105) are the only
# convolutions with a bias and no batch-norm, so their indices matter.
DETECTION_LAYERS = (81, 93, 105)


def _conv_block(inp, convs, skip=True):
    """A run of convolutions, optionally closed by a residual connection.

    Each entry of ``convs`` describes one convolution: ``filter``, ``kernel``,
    ``stride``, whether it is followed by batch-norm and leaky ReLU, and the
    Darknet layer index that fixes its position in the weights stream.
    """
    x = inp

    for count, conv in enumerate(convs):
        # The residual branch forks before the last two convolutions of a block.
        if count == (len(convs) - 2) and skip:
            skip_connection = x

        # Darknet pads asymmetrically (left and top only) when it downsamples,
        # which Keras' 'same' padding does not reproduce.
        if conv["stride"] > 1:
            x = ZeroPadding2D(((1, 0), (1, 0)))(x)

        x = Conv2D(
            conv["filter"],
            conv["kernel"],
            strides=conv["stride"],
            padding="valid" if conv["stride"] > 1 else "same",
            name="conv_" + str(conv["layer_idx"]),
            # A batch-norm layer has its own shift, so the convolution drops its bias.
            use_bias=not conv["bnorm"],
        )(x)

        if conv["bnorm"]:
            x = BatchNormalization(epsilon=0.001, name="bnorm_" + str(conv["layer_idx"]))(x)
        if conv["leaky"]:
            x = LeakyReLU(negative_slope=0.1, name="leaky_" + str(conv["layer_idx"]))(x)

    return Add()([skip_connection, x]) if skip else x


def build_yolov3(input_shape=(None, None, 3)):
    """Assemble YOLOv3: a Darknet-53 backbone plus a three-scale detection head.

    Returns a model with three outputs, one per scale. For a 416x416 input they
    are 13x13, 26x26 and 52x52, each 255 channels deep: 3 anchors x (4 box
    coordinates + 1 objectness + 80 COCO class scores).

    Leaving the spatial dimensions as ``None`` lets the same graph accept any
    input size that is a multiple of 32.
    """
    input_image = Input(shape=input_shape)

    # ---- Darknet-53 backbone -------------------------------------------------
    x = _conv_block(input_image, [
        {"filter": 32,  "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 0},
        {"filter": 64,  "kernel": 3, "stride": 2, "bnorm": True, "leaky": True, "layer_idx": 1},
        {"filter": 32,  "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 2},
        {"filter": 64,  "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 3}])

    x = _conv_block(x, [
        {"filter": 128, "kernel": 3, "stride": 2, "bnorm": True, "leaky": True, "layer_idx": 5},
        {"filter": 64,  "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 6},
        {"filter": 128, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 7}])

    x = _conv_block(x, [
        {"filter": 64,  "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 9},
        {"filter": 128, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 10}])

    x = _conv_block(x, [
        {"filter": 256, "kernel": 3, "stride": 2, "bnorm": True, "leaky": True, "layer_idx": 12},
        {"filter": 128, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 13},
        {"filter": 256, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 14}])

    for i in range(7):
        x = _conv_block(x, [
            {"filter": 128, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 16 + i * 3},
            {"filter": 256, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 17 + i * 3}])

    skip_36 = x   # 52x52 features, tapped for the finest detection scale

    x = _conv_block(x, [
        {"filter": 512, "kernel": 3, "stride": 2, "bnorm": True, "leaky": True, "layer_idx": 37},
        {"filter": 256, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 38},
        {"filter": 512, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 39}])

    for i in range(7):
        x = _conv_block(x, [
            {"filter": 256, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 41 + i * 3},
            {"filter": 512, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 42 + i * 3}])

    skip_61 = x   # 26x26 features, tapped for the middle detection scale

    x = _conv_block(x, [
        {"filter": 1024, "kernel": 3, "stride": 2, "bnorm": True, "leaky": True, "layer_idx": 62},
        {"filter": 512,  "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 63},
        {"filter": 1024, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 64}])

    for i in range(3):
        x = _conv_block(x, [
            {"filter": 512,  "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 66 + i * 3},
            {"filter": 1024, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 67 + i * 3}])

    # ---- Scale 1: 13x13, large objects --------------------------------------
    x = _conv_block(x, [
        {"filter": 512,  "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 75},
        {"filter": 1024, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 76},
        {"filter": 512,  "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 77},
        {"filter": 1024, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 78},
        {"filter": 512,  "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 79}],
        skip=False)

    yolo_82 = _conv_block(x, [
        {"filter": 1024, "kernel": 3, "stride": 1, "bnorm": True,  "leaky": True,  "layer_idx": 80},
        {"filter": 255,  "kernel": 1, "stride": 1, "bnorm": False, "leaky": False, "layer_idx": 81}],
        skip=False)

    # ---- Scale 2: 26x26, medium objects -------------------------------------
    x = _conv_block(x, [
        {"filter": 256, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 84}],
        skip=False)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, skip_61])

    x = _conv_block(x, [
        {"filter": 256, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 87},
        {"filter": 512, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 88},
        {"filter": 256, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 89},
        {"filter": 512, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 90},
        {"filter": 256, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 91}],
        skip=False)

    yolo_94 = _conv_block(x, [
        {"filter": 512, "kernel": 3, "stride": 1, "bnorm": True,  "leaky": True,  "layer_idx": 92},
        {"filter": 255, "kernel": 1, "stride": 1, "bnorm": False, "leaky": False, "layer_idx": 93}],
        skip=False)

    # ---- Scale 3: 52x52, small objects --------------------------------------
    x = _conv_block(x, [
        {"filter": 128, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 96}],
        skip=False)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, skip_36])

    yolo_106 = _conv_block(x, [
        {"filter": 128, "kernel": 1, "stride": 1, "bnorm": True,  "leaky": True,  "layer_idx": 99},
        {"filter": 256, "kernel": 3, "stride": 1, "bnorm": True,  "leaky": True,  "layer_idx": 100},
        {"filter": 128, "kernel": 1, "stride": 1, "bnorm": True,  "leaky": True,  "layer_idx": 101},
        {"filter": 256, "kernel": 3, "stride": 1, "bnorm": True,  "leaky": True,  "layer_idx": 102},
        {"filter": 128, "kernel": 1, "stride": 1, "bnorm": True,  "leaky": True,  "layer_idx": 103},
        {"filter": 256, "kernel": 3, "stride": 1, "bnorm": True,  "leaky": True,  "layer_idx": 104},
        {"filter": 255, "kernel": 1, "stride": 1, "bnorm": False, "leaky": False, "layer_idx": 105}],
        skip=False)

    return Model(input_image, [yolo_82, yolo_94, yolo_106], name="yolov3")


class WeightReader:
    """Sequential reader over a Darknet ``.weights`` file.

    Layout: three int32 version numbers, then an images-seen counter (int64 from
    format 2 onward, int32 before), then every parameter as float32 in layer
    order.
    """

    def __init__(self, weight_file):
        with open(weight_file, "rb") as handle:
            major, minor, revision = np.fromfile(handle, dtype=np.int32, count=3)

            if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
                np.fromfile(handle, dtype=np.int64, count=1)   # images seen
            else:
                np.fromfile(handle, dtype=np.int32, count=1)

            self.offset = 0
            self.all_weights = np.fromfile(handle, dtype=np.float32)

        self.version = (int(major), int(minor), int(revision))

    def read_bytes(self, size):
        self.offset += size
        if self.offset > len(self.all_weights):
            raise ValueError(
                f"Weights file exhausted at offset {self.offset} of "
                f"{len(self.all_weights)} — is this really a YOLOv3 .weights file?"
            )
        return self.all_weights[self.offset - size:self.offset]

    def remaining(self):
        return len(self.all_weights) - self.offset


def load_darknet_weights(model, weight_file, verbose=True):
    """Copy Darknet weights into a model built by :func:`build_yolov3`.

    Darknet writes, per convolution: the batch-norm parameters as
    ``beta, gamma, mean, variance`` (or a plain bias if there is no batch-norm),
    followed by the kernel. Keras wants ``gamma, beta, mean, variance``, and
    stores kernels as ``(h, w, in, out)`` where Darknet uses ``(out, in, h, w)``
    — so both need reordering.
    """
    reader = WeightReader(weight_file)
    loaded = 0

    for i in range(106):
        try:
            conv_layer = model.get_layer("conv_" + str(i))
        except ValueError:
            continue   # not every Darknet index is a convolution

        if i not in DETECTION_LAYERS:
            norm_layer = model.get_layer("bnorm_" + str(i))
            size = np.prod(norm_layer.get_weights()[0].shape)

            beta = reader.read_bytes(size)
            gamma = reader.read_bytes(size)
            mean = reader.read_bytes(size)
            var = reader.read_bytes(size)
            norm_layer.set_weights([gamma, beta, mean, var])

        if len(conv_layer.get_weights()) > 1:
            # Detection heads: bias first, then the kernel.
            bias = reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
            kernel = reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2, 3, 1, 0])
            conv_layer.set_weights([kernel, bias])
        else:
            kernel = reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2, 3, 1, 0])
            conv_layer.set_weights([kernel])

        loaded += 1

    leftover = reader.remaining()
    if leftover != 0:
        raise ValueError(
            f"{leftover} float32 values left unread after {loaded} convolutions. "
            "The architecture and the weights file do not agree."
        )

    if verbose:
        print(f"Loaded {loaded} convolutions from Darknet v{'.'.join(map(str, reader.version))}")
    return model
