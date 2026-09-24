# Detailed results

Every number here comes from a recorded training run on a T4 GPU, not from a
later re-run or an estimate.

## 1. Vehicle classifier

**Task.** 3-way classification of 32×32 RGB crops into `background`, `car`,
`truck`. 15,000 training images and 3,000 test images, balanced 5,000/1,000 per
class. Chance = 33.3%.

**Preprocessing.** Pixels scaled to [0, 1] then standardised per channel with
CIFAR's mean `(0.4914, 0.4822, 0.4465)` and std `(0.2023, 0.1994, 0.2010)`.
Labels one-hot encoded.

### Perceptron

`Input(32,32,3) → Flatten → Dense(128, relu) → Dense(3, softmax)`
Optimiser SGD, lr 1e-3, momentum 0.9, loss `categorical_crossentropy`.

| Run | Epochs | Final train acc | Final val acc | Best val acc |
|---|---|---|---|---|
| Short | 10 | 88.4% | 70.2% | 72.5% (epoch 7) |
| Long  | 20 | 96.5% | 70.9% | 72.5% (epoch 4) |

Validation accuracy plateaus around epoch 3 and never improves after; the
remaining epochs only widen the train/validation gap.

**Confusion matrix** (10-epoch run, accuracy 70.17%):

| True ↓ / Predicted → | Background | Car | Truck |
|---|---|---|---|
| **Background** | **781** | 75 | 144 |
| **Car** | 101 | **630** | 269 |
| **Truck** | 101 | 159 | **740** |

Per-class recall: background 78.1%, car 63.0%, truck 74.0%. The dominant error
is car→truck (269 cases, 26.9% of all cars). Background is the most reliable
class, which makes sense — it's the only one that isn't a boxy vehicle.

### CNN

`Input(32,32,3) → Conv2D(32, 3×3) → ReLU → MaxPool(2×2) → Flatten → Dense(128, relu) → Dense(3, softmax)`
Optimiser SGD, lr 1e-3, momentum 0.95.

| Epochs | Final train acc | Final val acc | Best val acc |
|---|---|---|---|
| 20 | 100.0% | 84.7% | 84.7% (epoch 14) |

A single convolutional layer buys **+13.8 points** over the perceptron. Training
accuracy hits 100% by epoch 17 while validation loss climbs from 0.43 to 0.64 —
textbook overfitting, and the reason for checkpointing on validation loss.

### VGG16 transfer learning

ImageNet-pretrained VGG16 (`include_top=False`) →
`GlobalAveragePooling2D → Dense(128, relu) → Dropout(0.3) → Dense(64, relu) → Dense(3, softmax)`
Optimiser SGD, lr 1e-3, momentum 0.9. All backbone layers trainable.

| Epochs | Final train acc | Final val acc | Best val acc |
|---|---|---|---|
| 20 | 99.8% | 94.6% | 94.6% (epoch 20) |

**+9.9 points over the CNN, +23.7 over the perceptron.** It clears 88.7%
validation accuracy after a *single* epoch — better than the CNN ever reaches in
20. That gap is the whole argument for transfer learning: the features that
distinguish a car from a truck were already learned from 14 million ImageNet
images, and only the 3-class head has to be fitted.

### Summary

| Model | Test accuracy | Δ vs previous |
|---|---|---|
| Chance | 33.3% | — |
| Perceptron | 70.9% | +37.6 |
| CNN | 84.7% | +13.8 |
| VGG16 transfer | 94.6% | +9.9 |

## 2. Sliding-window detection

**Setup.** A 100×160 street scene, scanned with a 32×32 window at 16px steps in
both directions, yielding 45 windows. Hand-labelled ground truth: 33 background
windows, 12 car windows, 0 truck windows.

| Model | Window accuracy |
|---|---|
| Perceptron (threshold 0.6) | 82.2% |
| CNN (threshold 0.9) | 82.2% — see caveat |

At a 0.6 confidence threshold the perceptron flagged 8 windows as vehicles, with
confidences from 0.67 to 0.99997. Two were labelled `truck` despite there being
no trucks in the scene — the same car/truck confusion the matrix above shows,
appearing in a detection setting.

> ### Caveat: the two numbers are identical for a reason
>
> The original evaluation helper took a `model` argument but called
> `perceptron.predict(...)` in its body, so the `model` argument was never used
> and scoring the CNN silently re-scored the perceptron — hence two identical
> 0.8222 results.
>
> That is fixed in `src/roadvision/sliding_window.py`, where `window_accuracy()`
> uses the model it is given. **The CNN's true sliding-window accuracy is
> therefore still unknown and needs a re-run.**

## 3. YOLOv3

**Setup.** Pretrained DarkNet-53, 416×416 letterboxed input, 80 COCO classes,
`obj_thresh=0.4`, `nms_thresh=0.45`.

Network output shapes for one image — three detection scales:

| Scale | Output shape | Grid | Targets |
|---|---|---|---|
| 1 | `(1, 13, 13, 255)` | 13×13 | Large objects |
| 2 | `(1, 26, 26, 255)` | 26×26 | Medium objects |
| 3 | `(1, 52, 52, 255)` | 52×52 | Small objects |

Depth 255 = 3 anchors × (4 box coordinates + 1 objectness + 80 class scores).

### Still image

On the street scene (720×400): **22 candidate boxes → 9 after NMS.**

| Class | Confidence | Box (left, top) → (right, bottom) |
|---|---|---|
| car | 1.00 | (653, 189) → (720, 243) |
| car | 0.99 | (625, 187) → (688, 233) |
| car | 0.98 | (462, 180) → (635, 256) |
| car | 0.48 | (375, 181) → (534, 243) |
| truck | 0.95 | (0, 161) → (268, 278) |
| bus | 0.91 | (353, 153) → (555, 233) |
| person | 0.93 | (297, 185) → (328, 253) |
| person | 0.88 | (325, 189) → (348, 258) |
| traffic light | 0.43 | (224, 123) → (239, 153) |

NMS removed 13 duplicates — for instance, six overlapping `car` boxes on the
right-hand vehicles collapsed to two, and the white pickup's two `truck` boxes
(0.95 and 0.85) collapsed to one. The before/after images are in
`docs/images/`.

### Video

A 300-frame dashcam clip, detected frame by frame. Typical per-frame output:
5–11 detections, mostly `car` plus `traffic light` and occasional `person`.
Inference ran at roughly 36–57 ms per frame on a T4 GPU, so ~20–27 fps for the
network pass alone, before the Python-side decoding and drawing.

Frames are processed independently with no tracking, so a vehicle's confidence
fluctuates between frames (e.g. the same car reading 0.98, 0.99, 0.98 across
three consecutive frames) and one spurious full-width box appeared on frame 2 at
0.51 confidence.

The rendered output is [`VideoResult.mp4`](../VideoResult.mp4) in the
repository root.

## What would be worth doing next

1. **Re-run the CNN sliding-window evaluation** with the fixed helper, so the
   CNN has a real number rather than the perceptron's.
2. **Report detection metrics, not classification accuracy.** mAP@0.5 against
   ground-truth boxes would make the YOLO results comparable to anything else.
3. **Add data augmentation** (flips, crops, colour jitter) to the classifier —
   with training accuracy at 100% and test at 85%, the CNN is limited by
   overfitting, not capacity.
4. **Fine-tune YOLO on road data** instead of using stock COCO weights.
5. **Add tracking** (SORT or ByteTrack) so detections persist across frames and
   the flicker disappears.

## Appendix: reproducing the classifier numbers

The figures above come from the original GPU run. Re-running
`scripts/train_classifier.py` from scratch on CPU, with no seeding, lands within
half a point of every one of them:

| Model | Metric | Original | Re-run | Delta |
|---|---|---|---|---|
| Perceptron | final train | 96.50% | 96.37% | −0.13 |
| Perceptron | final test | 70.90% | 71.33% | +0.43 |
| Perceptron | best test | 72.50% | 72.80% | +0.30 |
| CNN | final train | 100.00% | 100.00% | 0.00 |
| CNN | final test | 84.70% | 84.33% | −0.37 |
| CNN | best test | 84.73% | 84.67% | −0.06 |

Both runs are 20 epochs on the same balanced 15,000/3,000 split. The spread is
ordinary run-to-run variance from weight initialisation and shuffling; nothing
here depends on a lucky seed. Reproduce it with:

```bash
python scripts/train_classifier.py --model perceptron --epochs 20
python scripts/train_classifier.py --model cnn --epochs 20
```

The VGG16 transfer run is not re-verified here — it wants a GPU to be practical.

## Appendix: verifying the Darknet conversion

The weights ship as a Darknet `.weights` file — a 20-byte header followed by a
flat stream of float32 values, with no architecture and no layer names. Getting
the Keras rebuild wrong anywhere would silently produce a model that loads but
detects nothing, so it is checked arithmetically rather than by eye:

| Check | Expected | Result |
|---|---|---|
| Convolutions | 75 | 75 |
| Detection heads (the only unbiased convs) | 81, 93, 105 | 81, 93, 105 |
| Stride-2 downsamples | 5 | 5 |
| Output grids at 416×416 | 13, 26, 52 | 13, 26, 52 |
| Output depth | 255 | 255 |
| **Total parameters** | **62,001,757** | **62,001,757** |

That last row is the decisive one. 62,001,757 × 4 bytes + 20 bytes of header is
248,007,048 — exactly the size of the official `yolov3.weights`. Every float in
the file maps to one parameter in the graph, so a single wrong filter count
anywhere would break the equality.

`tests/test_models.py` then runs the loader against a synthetic file of that
exact shape and requires it to finish with nothing left unread, which pins the
architecture, the header parsing and the per-layer read order together. A short
file or a long one both raise.
