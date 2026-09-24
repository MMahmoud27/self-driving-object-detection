"""Run YOLOv3 over an image or a video.

    python scripts/detect.py street.jpg  --output outputs/street.png
    python scripts/detect.py dashcam.mp4 --output outputs/dashcam.mp4

The input type is picked from the file extension. Fetch the weights first:

    python scripts/download_assets.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PIL import Image

from roadvision.yolo import detect_image, detect_video, load_darknet

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
DEFAULT_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "data/yolo_weights.h5")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Image or video to run detection on")
    parser.add_argument("--output", required=True, help="Where to write the result")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--obj-thresh", type=float, default=0.4,
                        help="Minimum objectness for a box to count (default 0.4)")
    parser.add_argument("--nms-thresh", type=float, default=0.45,
                        help="Overlap above which a weaker box is dropped (default 0.45)")
    args = parser.parse_args()

    # Check the input before the weights: a typo in the filename is the more
    # likely mistake, and loading a 237 MB model first to then fail is wasteful.
    if not os.path.exists(args.input):
        parser.error(f"Input file not found: {args.input!r}")

    if not os.path.exists(args.weights):
        parser.error(
            f"Weights not found at {args.weights!r}. "
            "Run python scripts/download_assets.py first."
        )

    print(f"Loading YOLOv3 from {args.weights} ...")
    darknet = load_darknet(args.weights)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    extension = os.path.splitext(args.input)[1].lower()

    if extension in VIDEO_EXTENSIONS:
        detect_video(args.input, args.output, darknet,
                     obj_thresh=args.obj_thresh, nms_thresh=args.nms_thresh)
    else:
        image = Image.open(args.input).convert("RGB")
        detected = detect_image(image, darknet, obj_thresh=args.obj_thresh,
                                nms_thresh=args.nms_thresh, verbose=True)
        detected.save(args.output)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
