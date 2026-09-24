# Presentation

The slide deck is at
[`presentation/object-detection-for-self-driving-cars.pptx`](presentation/object-detection-for-self-driving-cars.pptx)
(15 slides, presented by Mahmoud, Viraj, Shloak and Jonathan).

## The video result

Slide 13 links out to the detection video rather than embedding it:

**<https://drive.google.com/file/d/17PSLYgAJQP6l5GMWqCWJdCPD0yb2KRgU/view>**

The `.pptx` contains only a hyperlink and a placeholder thumbnail — the video
file itself is not inside the deck and so is not in this repository. To put a
copy here, download it from that link and either commit it to `docs/` (if it's
small enough) or attach it to a GitHub release.

You can also regenerate it:

```bash
python scripts/download_assets.py
python scripts/detect.py data/video1.mp4 --output outputs/detected.mp4
```

## Slide outline

| # | Slide | Content |
|---|---|---|
| 1 | Title | Object Detection for Self-Driving Cars |
| 2 | How self-driving cars see | LiDAR, radar and camera vision; where classification fits |
| 3 | The data | CIFAR-10 subset: car, truck, background |
| 4 | Preprocessing and model design | Normalisation, one-hot labels, 128 hidden units |
| 5 | Results | ~70% accuracy; car/truck confusion; background easiest |
| 6 | What is a sliding window? | Scanning the image in overlapping patches |
| 7 | Sliding window code | The nested-loop implementation |
| 8 | Perceptron vs CNN | ~73% vs ~82% on sliding windows |
| 9 | The next step: transfer learning | Inexperienced models vs expert models |
| 10 | VGG16 | 14M images, 20,000+ categories, 95% on our three classes |
| 11 | The YOLO model | Grid → boxes + confidence → final detections |
| 12 | Non-maximal suppression | `obj_thresh`, `nms_thresh`, before/after |
| 13 | Now we turn it into a video | Frame-by-frame detection → the video above |
| 14 | Conclusion | |
| 15 | The End | |

## Where the deck and the notebooks disagree

Slide 8 reports **perceptron ~73% / CNN ~82%** on sliding windows. The recorded
notebook run gives **82.2% for both**, because of the `sliding_predictions` bug
described in [RESULTS.md](RESULTS.md#2-sliding-window-detection). If you present
this again, re-run the evaluation with the fixed helper in
`src/roadvision/sliding_window.py` and quote the new numbers.

Slide 4 says the optimiser was **Adam**; the notebooks use
**SGD(learning_rate=1e-3, momentum=0.9)** throughout. The code is the source of
truth here.
