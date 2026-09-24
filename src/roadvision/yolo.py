"""YOLOv3 inference: preprocessing, output decoding, NMS and drawing.

The network itself (DarkNet-53) is loaded from a pretrained Keras file; nothing
here trains it. What this module supplies is everything around the forward pass:
letterboxing the input to 416x416, turning three raw feature maps into boxes in
original-image coordinates, suppressing duplicate boxes, and rendering the
result.

DarkNet emits detections at three scales (13x13, 26x26 and 52x52 grids) so that
one network can find both a lorry filling the frame and a distant traffic light.
Each grid cell predicts 3 boxes, and each box carries 4 geometry values, 1
objectness score and 80 class scores -- hence the 255-deep output tensors.
"""

import colorsys
from copy import deepcopy

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# The 80 COCO classes YOLOv3 was trained on, in the order the network outputs.
COCO_LABELS = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# Prior box shapes in network pixels, grouped coarse -> fine to match the three
# output scales. They come from clustering the COCO training boxes, so the
# network predicts a small correction to a plausible shape rather than a size
# from scratch.
ANCHORS = [
    [[116, 90], [156, 198], [373, 326]],  # 13x13 grid, large objects
    [[30, 61], [62, 45], [59, 119]],      # 26x26 grid, medium objects
    [[10, 13], [16, 30], [33, 23]],       # 52x52 grid, small objects
]

NET_H = NET_W = 416


class BoundBox:
    """One predicted box plus its per-class scores."""

    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        return self.score


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _interval_overlap(interval_a, interval_b):
    """Length of the 1-D overlap between two intervals."""
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        return 0 if x4 < x1 else min(x2, x4) - x1
    return 0 if x2 < x3 else min(x2, x4) - x3


def bbox_iou(box1, box2):
    """Intersection over union: how much two boxes agree, from 0 to 1."""
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union if union else 0.0


def preprocess_input(image_pil, net_h=NET_H, net_w=NET_W):
    """Letterbox an image into the network's fixed 416x416 input.

    The image is scaled to fit while keeping its aspect ratio, then centred on a
    neutral grey canvas. Stretching instead would distort every object's shape
    and cost accuracy.
    """
    image = np.asarray(image_pil)
    # Drop an alpha channel if the upload has one; the network expects 3 channels.
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    new_h, new_w, _ = image.shape
    if (float(net_w) / new_w) < (float(net_h) / new_h):
        new_h = (new_h * net_w) / new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h) / new_h
        new_h = net_h
    new_w, new_h = int(new_w), int(new_h)

    resized = cv2.resize(image / 255.0, (new_w, new_h))

    canvas = np.ones((net_h, net_w, 3)) * 0.5
    canvas[
        int((net_h - new_h) // 2):int((net_h + new_h) // 2),
        int((net_w - new_w) // 2):int((net_w + new_w) // 2),
        :,
    ] = resized
    return np.expand_dims(canvas, 0)


def correct_yolo_boxes(boxes_, image_h, image_w, net_h=NET_H, net_w=NET_W):
    """Undo the letterbox, mapping boxes back to original-image pixels."""
    boxes = deepcopy(boxes_)
    if (float(net_w) / image_w) < (float(net_h) / image_h):
        new_w = net_w
        new_h = (image_h * net_w) / image_w
    else:
        new_h = net_w
        new_w = (image_w * net_h) / image_h

    for box in boxes:
        x_offset, x_scale = (net_w - new_w) / 2.0 / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2.0 / net_h, float(new_h) / net_h

        box.xmin = int((box.xmin - x_offset) / x_scale * image_w)
        box.xmax = int((box.xmax - x_offset) / x_scale * image_w)
        box.ymin = int((box.ymin - y_offset) / y_scale * image_h)
        box.ymax = int((box.ymax - y_offset) / y_scale * image_h)
    return boxes


def decode_netout(netout_, obj_thresh, anchors_, image_h, image_w,
                  net_h=NET_H, net_w=NET_W):
    """Turn the three raw feature maps into a list of candidate boxes.

    For each cell and anchor the network predicts offsets rather than absolute
    coordinates: the centre is a sigmoid offset within its own cell, and the
    size is an exponential scaling of the anchor's shape.
    """
    netout_all = deepcopy(netout_)
    boxes_all = []

    for scale_idx in range(len(netout_all)):
        netout = netout_all[scale_idx][0]
        anchors = anchors_[scale_idx]

        grid_h, grid_w = netout.shape[:2]
        nb_box = 3
        netout = netout.reshape((grid_h, grid_w, nb_box, -1))

        netout[..., :2] = _sigmoid(netout[..., :2])
        netout[..., 4:] = _sigmoid(netout[..., 4:])
        # Weight each class score by objectness, then zero out anything weak, so
        # a confident class on an empty cell cannot survive.
        netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
        netout[..., 5:] *= netout[..., 5:] > obj_thresh

        boxes = []
        for i in range(grid_h * grid_w):
            row, col = i // grid_w, i % grid_w

            for b in range(nb_box):
                objectness = netout[row][col][b][4]
                classes = netout[row][col][b][5:]

                if (classes <= obj_thresh).all():
                    continue

                x, y, w, h = netout[row][col][b][:4]
                x = (col + x) / grid_w                    # centre, fraction of width
                y = (row + y) / grid_h                    # centre, fraction of height
                w = anchors[b][0] * np.exp(w) / net_w     # width, fraction of width
                h = anchors[b][1] * np.exp(h) / net_h     # height, fraction of height

                boxes.append(
                    BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2,
                             objectness, classes)
                )

        boxes_all += boxes

    return correct_yolo_boxes(boxes_all, image_h, image_w, net_h, net_w)


def do_nms(boxes_, nms_thresh, obj_thresh):
    """Non-maximal suppression: one box per object, not fifteen.

    Per class, boxes are sorted by score and any weaker box overlapping a
    stronger one by more than ``nms_thresh`` has that class score zeroed. What
    survives above ``obj_thresh`` is returned.

    A box whose scores clear the threshold for two classes at once is emitted
    once, labelled with its highest scoring class, rather than once per
    qualifying class — which would draw an identical box several times over.
    """
    boxes = deepcopy(boxes_)
    if not boxes:
        return []

    num_class = len(boxes[0].classes)

    for c in range(num_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0:
                continue

            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0

    survivors = []
    for box in boxes:
        best = int(np.argmax(box.classes))
        if box.classes[best] > obj_thresh:
            box.label = best
            box.score = box.classes[best]
            survivors.append(box)

    return survivors


def _class_colors(num_labels):
    """One distinct colour per class, shuffled so neighbours look different."""
    hsv_tuples = [(x / num_labels, 1.0, 1.0) for x in range(num_labels)]
    colors = [colorsys.hsv_to_rgb(*x) for x in hsv_tuples]
    colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]

    np.random.seed(10101)   # fixed seed: a class keeps the same colour every run
    np.random.shuffle(colors)
    np.random.seed(None)
    return colors


def _load_font(image_h):
    """A font scaled to the image, falling back to PIL's default."""
    size = int(np.floor(3e-2 * image_h + 0.5))
    for path in (
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "C:/Windows/Fonts/consola.ttf",
    ):
        try:
            return ImageFont.truetype(font=path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_boxes(image_, boxes, labels=COCO_LABELS, verbose=False):
    """Draw labelled boxes onto a copy of ``image_`` and return it."""
    image = image_.copy()
    image_w, image_h = image.size
    font = _load_font(image_h)
    thickness = max(1, (image_w + image_h) // 300)
    colors = _class_colors(len(labels))

    for box in reversed(boxes):
        c = box.get_label()
        label = f"{labels[c]} {box.get_score():.2f}"

        top = max(0, int(np.floor(box.ymin + 0.5)))
        left = max(0, int(np.floor(box.xmin + 0.5)))
        bottom = min(image_h, int(np.floor(box.ymax + 0.5)))
        right = min(image_w, int(np.floor(box.xmax + 0.5)))

        # A box clipped entirely off-frame has nothing left to draw.
        if right <= left or bottom <= top:
            continue

        draw = ImageDraw.Draw(image)
        text_box = draw.textbbox((0, 0), label, font=font)
        label_size = (text_box[2], text_box[3])

        # Put the caption above the box, or just inside it when there is no room.
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        draw.rectangle([left, top, right, bottom], outline=colors[c], width=thickness)
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)],
                       fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

        if verbose:
            print(label, (left, top), (right, bottom))

    return image


def load_darknet(model_path, input_shape=(NET_H, NET_W, 3)):
    """Load YOLOv3 from either a saved Keras model or raw Darknet weights.

    A ``.weights`` file carries no architecture, so the graph is rebuilt first
    and the stream read into it; anything else is treated as a saved Keras
    model. ``compile=False`` skips reconstructing the training-time loss and
    optimiser, which inference does not need.
    """
    if str(model_path).endswith(".weights"):
        from roadvision.darknet import build_yolov3, load_darknet_weights

        model = build_yolov3(input_shape=input_shape)
        return load_darknet_weights(model, model_path)

    import tensorflow as tf

    return tf.keras.models.load_model(model_path, compile=False)


def detect_image(image_pil, darknet, obj_thresh=0.4, nms_thresh=0.45,
                 net_h=NET_H, net_w=NET_W, anchors=ANCHORS, labels=COCO_LABELS,
                 verbose=False):
    """Full single-image pipeline, returning an annotated PIL image."""
    image_w, image_h = image_pil.size

    new_image = preprocess_input(image_pil, net_h, net_w)
    yolo_outputs = darknet.predict(new_image, verbose=0)
    boxes = decode_netout(yolo_outputs, obj_thresh, anchors,
                          image_h, image_w, net_h, net_w)
    boxes = do_nms(boxes, nms_thresh, obj_thresh)

    return draw_boxes(image_pil, boxes, labels, verbose=verbose)


def detect_video(video_path, output_path, darknet, obj_thresh=0.4,
                 nms_thresh=0.45, net_h=NET_H, net_w=NET_W, anchors=ANCHORS,
                 labels=COCO_LABELS, progress_every=25):
    """Run detection over every frame of a video and write a new one.

    Frames are treated independently -- there is no tracking between them, which
    is why boxes can flicker on a hard frame.

    OpenCV works in BGR and PIL in RGB, so each frame is converted on the way in
    and back again on the way out.
    """
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise OSError(f"Couldn't open video: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = vid.get(cv2.CAP_PROP_FPS)
    size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    num_frame = 0
    try:
        while vid.isOpened():
            ret, frame = vid.read()
            if not ret:
                break

            num_frame += 1
            if progress_every and num_frame % progress_every == 0:
                print(f"... frame {num_frame}")

            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detected = detect_image(frame_pil, darknet, obj_thresh, nms_thresh,
                                    net_h, net_w, anchors, labels)
            out.write(cv2.cvtColor(np.asarray(detected), cv2.COLOR_RGB2BGR))
    finally:
        vid.release()
        out.release()

    print(f"Wrote {num_frame} frames to {output_path}")
    return output_path
