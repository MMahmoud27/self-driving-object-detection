"""Box geometry and non-maximal suppression.

These exercise the pure-numpy half of the detector, so they run without
TensorFlow and without the 237 MB weights file.
"""

import numpy as np
import pytest

from roadvision.yolo import BoundBox, bbox_iou, do_nms

CAR, TRUCK = 2, 7   # COCO class indices


def box(x1, y1, x2, y2, score, cls=CAR, num_classes=80):
    classes = np.zeros(num_classes)
    classes[cls] = score
    return BoundBox(x1, y1, x2, y2, score, classes)


class TestIoU:
    def test_identical_boxes_overlap_completely(self):
        a, b = box(0, 0, 10, 10, 0.9), box(0, 0, 10, 10, 0.9)
        assert bbox_iou(a, b) == pytest.approx(1.0)

    def test_disjoint_boxes_do_not_overlap(self):
        a, b = box(0, 0, 10, 10, 0.9), box(50, 50, 60, 60, 0.9)
        assert bbox_iou(a, b) == 0.0

    def test_partial_overlap(self):
        # Two 10x10 boxes offset by 5 in both axes: intersection 25, union 175.
        a, b = box(0, 0, 10, 10, 0.9), box(5, 5, 15, 15, 0.8)
        assert bbox_iou(a, b) == pytest.approx(25 / 175)

    def test_contained_box(self):
        # A 5x5 box wholly inside a 10x10 one: intersection 25, union 100.
        a, b = box(0, 0, 10, 10, 0.9), box(2, 2, 7, 7, 0.8)
        assert bbox_iou(a, b) == pytest.approx(25 / 100)

    def test_touching_edges_is_not_overlap(self):
        a, b = box(0, 0, 10, 10, 0.9), box(10, 0, 20, 10, 0.9)
        assert bbox_iou(a, b) == 0.0

    def test_is_symmetric(self):
        a, b = box(0, 0, 10, 10, 0.9), box(3, 4, 14, 12, 0.8)
        assert bbox_iou(a, b) == pytest.approx(bbox_iou(b, a))


class TestNMS:
    def test_empty_input(self):
        assert do_nms([], 0.45, 0.4) == []

    def test_single_box_survives(self):
        assert len(do_nms([box(0, 0, 10, 10, 0.9)], 0.45, 0.4)) == 1

    def test_duplicates_collapse_to_the_strongest(self):
        boxes = [box(100, 100, 200, 200, 0.95),
                 box(105, 103, 203, 198, 0.80),
                 box(98, 99, 197, 201, 0.60)]
        kept = do_nms(boxes, 0.45, 0.4)
        assert len(kept) == 1
        assert kept[0].get_score() == pytest.approx(0.95)

    def test_distinct_objects_are_both_kept(self):
        boxes = [box(100, 100, 200, 200, 0.95), box(400, 100, 500, 200, 0.90)]
        assert len(do_nms(boxes, 0.45, 0.4)) == 2

    def test_below_threshold_is_dropped(self):
        assert do_nms([box(0, 0, 10, 10, 0.3)], 0.45, 0.4) == []

    def test_different_classes_do_not_suppress_each_other(self):
        # Suppression is per class, so an overlapping car and truck both stay.
        boxes = [box(100, 100, 200, 200, 0.95, cls=CAR),
                 box(102, 101, 198, 202, 0.90, cls=TRUCK)]
        assert len(do_nms(boxes, 0.45, 0.4)) == 2

    def test_box_scoring_two_classes_is_emitted_once(self):
        classes = np.zeros(80)
        classes[CAR] = 0.55
        classes[TRUCK] = 0.85
        kept = do_nms([BoundBox(10, 10, 50, 50, 0.9, classes)], 0.45, 0.4)
        assert len(kept) == 1
        assert kept[0].get_label() == TRUCK   # labelled with its best class

    def test_threshold_controls_aggressiveness(self):
        # IoU here is ~0.68: suppressed at 0.45, kept at 0.9.
        boxes = [box(0, 0, 100, 100, 0.95), box(15, 15, 115, 115, 0.85)]
        assert len(do_nms(boxes, 0.45, 0.4)) == 1
        assert len(do_nms(boxes, 0.90, 0.4)) == 2


class TestBoundBox:
    def test_label_is_the_argmax_class(self):
        assert box(0, 0, 1, 1, 0.9, cls=CAR).get_label() == CAR

    def test_score_matches_the_labelled_class(self):
        assert box(0, 0, 1, 1, 0.77, cls=TRUCK).get_score() == pytest.approx(0.77)

    def test_label_is_cached_after_first_call(self):
        b = box(0, 0, 1, 1, 0.9, cls=CAR)
        assert b.get_label() == CAR
        b.classes[TRUCK] = 0.99      # mutate after the fact
        assert b.get_label() == CAR  # memoised, not recomputed
