# Notice on authorship and reuse

**There is deliberately no open-source licence on this repository.** Without one,
the default applies: all rights reserved. You are welcome to read the code — that
is what publishing it on GitHub grants — but not to copy, modify or redistribute
it.

That is not caution for its own sake. This project is coursework built on top of
material I did not write, and an MIT or Apache licence would be a grant of
rights I do not hold.

## Who wrote what

This started as a guided project in the **Inspirit AI Scholars** program. The
notebooks came pre-built with the scaffolding already in place; students filled
in the cells marked `### YOUR CODE HERE`.

**Provided by the course** — everything inside the
`#@title Run this to download data and prepare our environment!` cells:

- The whole YOLOv3 helper block: `BoundBox`, `preprocess_input`,
  `decode_netout`, `correct_yolo_boxes`, `do_nms`, `bbox_iou`,
  `_interval_overlap`, `_sigmoid`, `draw_boxes`
- Dataset construction and preprocessing: `load_cifar10`,
  `construct_vehicle_dataset`, `load_vehicle_dataset`, `normalize`,
  `label_to_onehot`
- Plotting: `plot_one_image`, `plot_acc`, `plot_confusion_matrix`
- The transfer-learning wrapper `TransferClassifier_func`
- `launch_website`, and the notebook prose and exercise structure throughout
- The pretrained YOLOv3 weights, the sample images and the source video, all
  hosted in the course's storage bucket

**Written by Mahmoud Mohamed** — every exercise cell in the notebooks:

- The sliding-window extraction loop
- The bodies of `detect_image` and `detect_video`
- The model definitions built to the course's specification
- The window-classification and thresholding code
- The rendered detection video

**Added afterwards, outside the course** (see the git history):

- `src/roadvision/` — the course helpers and my exercise answers refactored into
  an importable package, deduplicated across notebooks and documented
- `app/app.py` — a working Streamlit front end. The notebook's own app cells
  were left unfinished during the course and have since been completed
- `scripts/` and the `docs/` write-ups

Because the derived package in `src/` is substantially a reorganisation of the
course's code, it carries the same restriction as the original.

## Third-party components

| Component | Source | Terms |
|---|---|---|
| YOLOv3 / DarkNet weights | Redmon & Farhadi; redistributed via the course bucket | Original release was public domain; the redistributed copy is the course's |
| VGG16 / ResNet50 / DenseNet121 weights | Keras Applications | Each carries its own upstream licence |
| CIFAR-10 | Krizhevsky, University of Toronto | Free for research use |
| Sample images and video | Course storage bucket | Unknown — not mine to relicense |
| Diagrams in the slide deck | YOLO paper, third-party blog posts, stock photography | Third-party; reproduced in the deck for a classroom presentation |

`scripts/download_assets.py` fetches the weights and media from the course
bucket at runtime rather than vendoring them here, so none of that material is
committed to this repository.

## If you want to reuse something

Ask. Open an issue, or contact the repository owner. For anything originating
with the course, permission has to come from Inspirit AI, not from me.
