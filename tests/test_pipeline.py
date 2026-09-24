"""Preprocessing, window extraction and output decoding.

Decoding is driven by a synthetic feature map rather than a real forward pass,
so the whole file runs in about a second with no weights and no TensorFlow.
"""

import numpy as np
import pytest
from PIL import Image

from roadvision.data import label_to_onehot, normalize
from roadvision.sliding_window import extract_windows
from roadvision.yolo import (
    ANCHORS,
    COCO_LABELS,
    correct_yolo_boxes,
    decode_netout,
    do_nms,
    draw_boxes,
    preprocess_input,
)

CAR = 2


@pytest.fixture
def scene():
    """A deterministic 100x160 RGB image."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (100, 160, 3), dtype=np.uint8)


class TestSlidingWindow:
    def test_window_count_and_shape(self, scene):
        windows, positions = extract_windows(scene)
        # 100x160 with a 32px window at 16px steps -> 5 rows x 9 cols
        assert windows.shape == (45, 32, 32, 3)
        assert len(positions) == 45

    def test_positions_track_the_grid(self, scene):
        _, positions = extract_windows(scene)
        assert positions[0] == (0, 0)
        assert positions[1] == (16, 0)
        assert positions[9] == (0, 16)

    def test_windows_match_the_source_pixels(self, scene):
        windows, positions = extract_windows(scene)
        for idx in (0, 17, 44):
            x, y = positions[idx]
            assert np.array_equal(windows[idx], scene[y:y + 32, x:x + 32])

    def test_partial_edge_windows_are_dropped(self, scene):
        windows, _ = extract_windows(scene, window_h=32, window_w=32,
                                     step_h=32, step_w=32)
        # 160/32 = 5 columns exactly; 100/32 = 3 full rows, the 4th is short.
        assert windows.shape == (15, 32, 32, 3)

    def test_step_equal_to_size_gives_no_overlap(self, scene):
        _, positions = extract_windows(scene, step_h=32, step_w=32)
        xs = sorted({x for x, _ in positions})
        assert xs == [0, 32, 64, 96, 128]


class TestPreprocessing:
    def test_normalize_matches_the_standardisation_formula(self):
        from roadvision.data import CIFAR_MEAN, CIFAR_STD

        grey = np.full((4, 4, 3), 127, dtype=np.uint8)
        expected = (127 / 255 - CIFAR_MEAN) / CIFAR_STD
        assert normalize(grey)[0, 0] == pytest.approx(expected)

    def test_normalize_centres_mid_grey_near_zero(self):
        # Mid-grey sits close to the dataset mean, so it lands near zero --
        # furthest on blue, whose mean (0.4465) is lowest of the three.
        grey = np.full((4, 4, 3), 127, dtype=np.uint8)
        assert np.abs(normalize(grey)).max() < 0.3

    def test_normalize_preserves_shape(self, scene):
        assert normalize(scene).shape == scene.shape

    def test_label_to_onehot(self):
        assert label_to_onehot([0, 1, 2, 1]).tolist() == [
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]
        ]

    def test_letterbox_output_shape(self, scene):
        assert preprocess_input(Image.fromarray(scene)).shape == (1, 416, 416, 3)

    def test_letterbox_pads_with_grey(self, scene):
        # A 160x100 image is wider than tall, so the padding lands top and bottom.
        out = preprocess_input(Image.fromarray(scene))[0]
        assert out[0, 208, 0] == pytest.approx(0.5)
        assert out[415, 208, 0] == pytest.approx(0.5)

    def test_letterbox_preserves_aspect_ratio(self):
        tall = Image.fromarray(np.zeros((400, 100, 3), dtype=np.uint8))
        out = preprocess_input(tall)[0]
        # 100x400 scaled to fit 416 high -> 104 wide, centred with grey either side.
        assert out[208, 0, 0] == pytest.approx(0.5)
        assert out[208, 415, 0] == pytest.approx(0.5)

    def test_alpha_channel_is_dropped(self):
        rgba = Image.fromarray(np.zeros((50, 50, 4), dtype=np.uint8), "RGBA")
        assert preprocess_input(rgba).shape == (1, 416, 416, 3)


def fake_netout(detections, grids=(13, 26, 52)):
    """Build three feature maps with detections planted at chosen cells.

    ``detections`` maps ``(scale_index, row, col)`` to a class index. Everything
    else is set well below the sigmoid's midpoint so it decodes as empty.
    """
    outs = [np.full((1, g, g, 255), -8.0, dtype=np.float32) for g in grids]
    for (scale, row, col), cls in detections.items():
        outs[scale][0, row, col, 0:2] = 0.0    # centre offset within the cell
        outs[scale][0, row, col, 2:4] = -0.5   # size, relative to the anchor
        outs[scale][0, row, col, 4] = 5.0      # objectness -> ~0.993
        outs[scale][0, row, col, 5 + cls] = 5.0
    return outs


class TestDecoding:
    def test_empty_output_yields_no_boxes(self):
        outs = fake_netout({})
        assert decode_netout(outs, 0.4, ANCHORS, 400, 720) == []

    def test_planted_detection_is_recovered(self):
        outs = fake_netout({(0, 6, 6): CAR})
        boxes = decode_netout(outs, 0.4, ANCHORS, 400, 720)
        assert len(boxes) == 1
        assert boxes[0].get_label() == CAR
        assert COCO_LABELS[boxes[0].get_label()] == "car"

    def test_box_lands_in_image_coordinates(self):
        outs = fake_netout({(0, 6, 6): CAR})
        box = decode_netout(outs, 0.4, ANCHORS, 400, 720)[0]
        # Cell (6,6) of a 13x13 grid is the middle of the frame.
        cx, cy = (box.xmin + box.xmax) / 2, (box.ymin + box.ymax) / 2
        assert 300 < cx < 420
        assert 160 < cy < 240
        assert box.xmin < box.xmax and box.ymin < box.ymax

    def test_detections_at_every_scale(self):
        outs = fake_netout({(0, 6, 6): CAR, (1, 13, 13): CAR, (2, 26, 26): CAR})
        assert len(decode_netout(outs, 0.4, ANCHORS, 400, 720)) == 3

    def test_finer_scales_produce_smaller_boxes(self):
        coarse = decode_netout(fake_netout({(0, 6, 6): CAR}), 0.4, ANCHORS, 400, 720)[0]
        fine = decode_netout(fake_netout({(2, 26, 26): CAR}), 0.4, ANCHORS, 400, 720)[0]
        assert (coarse.xmax - coarse.xmin) > (fine.xmax - fine.xmin)

    def test_threshold_filters_weak_detections(self):
        outs = fake_netout({(0, 6, 6): CAR})
        outs[0][0, 6, 6, 4] = -1.0            # objectness ~0.27
        assert decode_netout(outs, 0.4, ANCHORS, 400, 720) == []

    def test_decode_does_not_mutate_its_input(self):
        outs = fake_netout({(0, 6, 6): CAR})
        before = outs[0].copy()
        decode_netout(outs, 0.4, ANCHORS, 400, 720)
        assert np.array_equal(outs[0], before)


class TestDrawing:
    def test_returns_an_image_of_the_same_size(self, scene):
        image = Image.fromarray(scene)
        boxes = do_nms(decode_netout(fake_netout({(0, 6, 6): CAR}), 0.4,
                                     ANCHORS, 100, 160), 0.45, 0.4)
        out = draw_boxes(image, boxes)
        assert out.size == image.size

    def test_does_not_modify_the_original(self, scene):
        image = Image.fromarray(scene)
        before = np.asarray(image).copy()
        boxes = do_nms(decode_netout(fake_netout({(0, 6, 6): CAR}), 0.4,
                                     ANCHORS, 100, 160), 0.45, 0.4)
        draw_boxes(image, boxes)
        assert np.array_equal(np.asarray(image), before)

    def test_drawing_actually_changes_pixels(self, scene):
        image = Image.fromarray(scene)
        boxes = do_nms(decode_netout(fake_netout({(0, 6, 6): CAR}), 0.4,
                                     ANCHORS, 100, 160), 0.45, 0.4)
        out = draw_boxes(image, boxes)
        assert not np.array_equal(np.asarray(out), np.asarray(image))

    def test_no_boxes_leaves_the_image_alone(self, scene):
        image = Image.fromarray(scene)
        out = draw_boxes(image, [])
        assert np.array_equal(np.asarray(out), np.asarray(image))


class TestCoordinateCorrection:
    def test_correction_does_not_mutate_its_input(self):
        from roadvision.yolo import BoundBox

        boxes = [BoundBox(0.1, 0.1, 0.5, 0.5, 0.9, np.zeros(80))]
        correct_yolo_boxes(boxes, 400, 720)
        assert boxes[0].xmin == pytest.approx(0.1)   # original untouched
