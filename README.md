# Object Detection for Self-Driving Cars

Building up a road-scene object detector the long way round — starting from a
perceptron that can barely tell a car from a truck, and ending with YOLOv3
labelling every vehicle, pedestrian and traffic light in a dashcam video.

Each stage exists to expose the limitation that motivates the next one:

| Stage | Approach | What it fixed | What it couldn't do |
|---|---|---|---|
| 1 | Perceptron on 32×32 crops | Classifies `background` / `car` / `truck` | Flattening throws away spatial structure — **70% test accuracy** |
| 2 | CNN + sliding window | Convolutions keep spatial structure; sliding windows add *where* | One forward pass per window; boxes snap to a grid — **85%** |
| 3 | VGG16 transfer learning | ImageNet features beat learning from scratch on 15k images — **95%** | Still a classifier bolted onto a brute-force scan |
| 4 | YOLOv3 | One pass over the whole image, 80 classes, real boxes at 3 scales | No frame-to-frame tracking |

<p align="center">
  <img src="docs/images/yolo-before-nms.png" width="48%" alt="YOLOv3 raw output: 22 overlapping boxes">
  <img src="docs/images/yolo-after-nms.png" width="48%" alt="After non-maximal suppression: 9 clean boxes">
</p>
<p align="center"><em>YOLOv3 raw output (22 candidate boxes) → after non-maximal suppression (9 detections).</em></p>

---

## Results

### Vehicle classifier — 3 classes, 15,000 train / 3,000 test images

| Model | Epochs | Train accuracy | Test accuracy |
|---|---|---|---|
| Perceptron — `Flatten → Dense(128) → Dense(3)` | 10 | 88.4% | 70.2% |
| Perceptron | 20 | 96.5% | 70.9% |
| CNN — `Conv2D(32) → ReLU → MaxPool → Dense(128) → Dense(3)` | 20 | 100% | **84.7%** |
| VGG16 transfer — ImageNet backbone + custom head | 20 | 99.8% | **94.6%** |

Chance is 33.3% — the dataset is balanced by construction.

<p align="center">
  <img src="docs/images/training-accuracy.png" width="46%" alt="Training vs validation accuracy across epochs">
  <img src="docs/images/confusion-matrix.png" width="50%" alt="Confusion matrix for the perceptron">
</p>

Two things stand out. **The models overfit hard**: the perceptron's training
accuracy climbs past 96% while validation sits flat at ~71% from epoch 3
onwards, and the CNN reaches a perfect 100% on training data. Every run
therefore checkpoints on best validation loss rather than keeping the last
epoch.

**Cars and trucks are what get confused.** From the perceptron's confusion
matrix, 269 of 1,000 cars were called trucks and 159 of 1,000 trucks were called
cars, while background was the easiest class by a wide margin. At 32×32 pixels a
hatchback and a pickup really do look alike; that ambiguity is what the deeper
models buy their way out of.

### Sliding-window detection

Scanning a 100×160 street scene with a 32×32 window at 16px steps gives 45
windows, of which 12 contain a vehicle. The perceptron scored **82.2%** across
those windows.

> **Caveat on the CNN sliding-window number.** The notebook's
> `sliding_predictions(model, windows, threshold)` helper took a `model`
> argument but called `perceptron.predict()` in its body, so passing the CNN in
> silently re-scored the perceptron — both report exactly 0.8222. The
> presentation's "perceptron ~73% / CNN ~82%" comes from a separate run. The
> bug is fixed in [`src/roadvision/sliding_window.py`](src/roadvision/sliding_window.py);
> the CNN figure needs a re-run to be trustworthy.

### YOLOv3

On the street-scene test image, YOLOv3 produced 22 candidate boxes above the
0.4 objectness threshold, which NMS reduced to 9 detections: 4 cars, 2 people,
1 bus, 1 truck and 1 traffic light. Applied frame-by-frame to a 301-frame
dashcam clip, it tracked vehicles and traffic lights throughout.

📹 **The rendered result is in this repo: [`VideoResult.mp4`](VideoResult.mp4)**
(also [on Google Drive](https://drive.google.com/file/d/17PSLYgAJQP6l5GMWqCWJdCPD0yb2KRgU/view),
which is what slide 13 of the deck links to).

---

## Quickstart

```bash
git clone https://github.com/MMahmoud27/self-driving-object-detection.git
cd self-driving-object-detection
pip install -r requirements.txt
```

Fetch the pretrained YOLOv3 weights and sample media (~240 MB, one-off):

```bash
python scripts/download_assets.py
```

Run detection on an image or a video:

```bash
python scripts/detect.py data/image.jpg --output outputs/detected.png
```

```bash
python scripts/detect.py data/video1.mp4 --output outputs/detected.mp4
```

Train a classifier from scratch:

```bash
python scripts/train_classifier.py --model cnn --epochs 20
```

Launch the web app:

```bash
streamlit run app/app.py
```

The app takes an uploaded road scene and returns it annotated, with sliders for
the objectness and NMS thresholds so you can watch the precision/recall
trade-off directly.

---

## Repository layout

```
├── notebooks/          The original Colab notebooks, outputs intact
│   ├── 01_image_classification.ipynb
│   ├── 02_sliding_window_and_transfer_learning.ipynb
│   ├── 03_yolov3_detection.ipynb
│   └── 04_streamlit_deployment.ipynb
├── src/roadvision/     The same code, factored into an importable package
│   ├── data.py             CIFAR-10 → 3-class vehicle dataset, normalisation
│   ├── models.py           Perceptron, CNN, transfer-learning builders
│   ├── sliding_window.py   Window extraction and classification
│   ├── yolo.py             YOLOv3 pre/post-processing, image + video detection
│   └── viz.py              Accuracy curves, confusion matrices
├── app/app.py          Streamlit front end
├── scripts/            Command-line entry points
├── VideoResult.mp4     The rendered YOLOv3 detection video
└── docs/               Presentation, result images, detailed write-up
```

The notebooks are otherwise exactly as they ran on Colab, so the outputs saved in
them are the real recorded results.

> **Seven cells were completed after the course ended** and have no saved
> outputs, because they have not been re-run: notebook 02 cells 25 and 79–85
> (Activity 4b, VGG16 assembled by hand in Keras) and notebook 04 cells 26 and 30
> (`utils.py` and the finished `app.py`). Run them in Colab with a GPU to fill in
> their results. Every other cell's output is from the original session.

The notebooks also duplicate a lot of code — notebooks 3 and 4 each carry their
own copy of the same ~300-line YOLO helper block. `src/roadvision/` is that code
deduplicated and documented, and it is what the app and scripts import.

---

## How it works

### The dataset

CIFAR-10 has ten classes; only two are vehicles worth detecting. The project
keeps `automobile` and `truck`, and folds the other eight classes into a single
`background` class so the model learns "not a vehicle" as an explicit answer
rather than being forced to guess. 5,000 images per class for training, 1,000
for test — balanced, so accuracy is directly interpretable against a 33.3%
chance baseline.

### Sliding windows

A classifier answers *what is this?*; detection also needs *where is it?*. The
cheapest bridge is to slide a fixed 32×32 window across the image in 16px steps
— overlapping, so an object straddling a grid line still lands fully inside at
least one crop — and classify every crop. It works, and it is slow, size-locked
and grid-snapped. Those three limits are exactly what YOLO removes.

### YOLOv3

One network pass over the whole image produces detections at three scales
(13×13, 26×26 and 52×52 grids), so a lorry filling the frame and a distant
traffic light are both findable. Each grid cell predicts 3 boxes; each box
carries 4 geometry values, 1 objectness score and 80 class scores — which is why
the output tensors are 255 deep.

Two thresholds control the output:

- **`obj_thresh`** (default 0.4) — how sure the model must be before a box
  counts at all. Raise it to cut false positives, lower it to catch faint
  objects.
- **`nms_thresh`** (default 0.45) — how much two boxes may overlap before the
  weaker one is discarded. This is what turns the 22 raw boxes above into 9.

---

## Limitations

- **Frames are independent.** Video detection runs YOLO on each frame with no
  tracking between them, so boxes can flicker and an object briefly occluded is
  simply lost and re-found.
- **The custom classifier is trained at 32×32.** That resolution is why cars and
  trucks blur together; it was chosen to keep training fast, not because it's
  right for road scenes.
- **YOLOv3 here is pretrained on COCO, not fine-tuned.** It has never seen this
  project's data. It is also a 2018 model — YOLOv8/v11 are considerably faster
  and more accurate.
- **No real detection metrics.** Everything reported is classification accuracy
  or eyeballed boxes. Proper evaluation would need mAP against ground-truth
  bounding boxes, which this dataset doesn't provide.
- **Not remotely road-safe.** This is a teaching pipeline. Real autonomous
  vehicles fuse camera, LiDAR and radar, and treat a missed detection as a
  safety event rather than a percentage point.

---

## Credits

Code by **Mahmoud Mohamed**, written during the Inspirit AI Scholars program
and extended afterwards.

- YOLOv3 — Redmon & Farhadi, [*YOLOv3: An Incremental Improvement*](https://arxiv.org/abs/1804.02767); original paper [*You Only Look Once*](https://arxiv.org/abs/1506.02640)
- VGG16 — Simonyan & Zisserman, [*Very Deep Convolutional Networks*](https://arxiv.org/abs/1409.1556)
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — Krizhevsky, University of Toronto
- Pretrained weights and sample media come from the Inspirit AI course bucket

The notebooks arrived with their scaffolding and helper functions already
written; the team filled in the `### YOUR CODE HERE` cells. `src/roadvision/`,
`app/app.py` and `scripts/` were written afterwards, outside the course.
[NOTICE.md](NOTICE.md) has the full breakdown.

The presentation is in [`docs/presentation/`](docs/presentation/), and a fuller
write-up of the numbers is in [`docs/RESULTS.md`](docs/RESULTS.md).

## Licensing and reuse

**No open-source licence — all rights reserved.** Read it freely; please don't
copy, modify or redistribute it without asking.

That's deliberate. A large share of this code came with the course rather than
being written by the team, so an MIT or Apache licence would be granting rights
we don't hold. [NOTICE.md](NOTICE.md) sets out exactly which parts are the
course's, which are the team's, and who to ask about reuse.
