"""Fetch the official YOLOv3 weights and convert them for Keras.

    python scripts/download_assets.py

Downloads ``yolov3.weights`` (237 MB) from the original Darknet release,
rebuilds the network in Keras and saves it as ``data/yolo_weights.h5``. The
download only happens once; so does the conversion.

``data/`` is gitignored, so nothing here is committed.
"""

import argparse
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# The weights as published with the YOLOv3 paper.
WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3.weights"
EXPECTED_BYTES = 248_007_048

DATA_ROOT = "data"
DARKNET_WEIGHTS = os.path.join(DATA_ROOT, "yolov3.weights")
KERAS_WEIGHTS = os.path.join(DATA_ROOT, "yolo_weights.h5")


def _progress(count, block_size, total_size):
    if total_size <= 0:
        return
    downloaded = count * block_size
    percent = min(100, downloaded * 100 // total_size)
    bar = "=" * (percent // 3)
    sys.stdout.write(
        f"\r    [{bar:<33}] {percent:3d}%  "
        f"{downloaded / 1e6:6.1f} / {total_size / 1e6:.1f} MB"
    )
    sys.stdout.flush()


def download_weights(url=WEIGHTS_URL, destination=DARKNET_WEIGHTS):
    if os.path.exists(destination):
        print(f"[skip] {destination} already present "
              f"({os.path.getsize(destination) / 1e6:.1f} MB)")
        return destination

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    print(f"[get ] {url}")
    # Download beside the target and rename at the end, so an interrupted
    # transfer never leaves a half-written file that looks complete.
    partial = destination + ".part"
    urllib.request.urlretrieve(url, partial, reporthook=_progress)
    print()

    size = os.path.getsize(partial)
    if size != EXPECTED_BYTES:
        print(f"  warning: expected {EXPECTED_BYTES} bytes, got {size}")

    os.replace(partial, destination)
    return destination


def convert(darknet_weights=DARKNET_WEIGHTS, destination=KERAS_WEIGHTS):
    if os.path.exists(destination):
        print(f"[skip] {destination} already present")
        return destination

    # Imported here so the download path does not need TensorFlow loaded.
    from roadvision.darknet import build_yolov3, load_darknet_weights

    print("[conv] building the network and loading weights...")
    model = build_yolov3(input_shape=(416, 416, 3))
    load_darknet_weights(model, darknet_weights)

    model.save(destination)
    print(f"[save] {destination} ({os.path.getsize(destination) / 1e6:.1f} MB)")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-convert", action="store_true",
                        help="Download the .weights file but don't build the .h5")
    args = parser.parse_args()

    download_weights()
    if args.skip_convert:
        print("\nDone (conversion skipped).")
        return

    convert()
    print("\nDone. Run detection with:")
    print("    python scripts/detect.py <your-image-or-video> --output out.png")


if __name__ == "__main__":
    main()
