"""Fetch the YOLOv3 weights and the sample media used by the notebooks.

    python scripts/download_assets.py

Everything lands in ``data/``, which is gitignored. The weights file alone is
237 MB, so this is a one-off download rather than something committed.

These are the same URLs the Colab notebooks pull from (the Inspirit AI course
bucket). The CIFAR-10 data is not listed here -- Keras downloads it on demand
the first time ``load_vehicle_dataset()`` runs.
"""

import os
import sys
import urllib.request

BUCKET = (
    "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/"
    "Sessions%206%20-%2010%20(Projects)/"
    "Project%20-%20%20Object%20Detection%20(Autonomous%20Vehicles)"
)

ASSETS = {
    "yolo_weights.h5": f"{BUCKET}/yolo.h5",       # 237 MB pretrained DarkNet
    "image.jpg": f"{BUCKET}/image.jpg",           # street scene used in notebook 3
    "image2.jpg": f"{BUCKET}/image2.jpg",
    "image-2.jpg": f"{BUCKET}/image-2.jpg",       # 100x160 scene for sliding windows
    "video1.mp4": f"{BUCKET}/6.mp4",              # 301-frame dashcam clip
}

DATA_ROOT = "data"


def _progress(count, block_size, total_size):
    if total_size <= 0:
        return
    downloaded = count * block_size
    percent = min(100, downloaded * 100 // total_size)
    sys.stdout.write(
        f"\r    {percent:3d}%  ({downloaded / 1e6:.1f} / {total_size / 1e6:.1f} MB)"
    )
    sys.stdout.flush()


def main():
    os.makedirs(DATA_ROOT, exist_ok=True)

    for filename, url in ASSETS.items():
        destination = os.path.join(DATA_ROOT, filename)

        if os.path.exists(destination):
            print(f"[skip] {filename} already present")
            continue

        print(f"[get ] {filename}")
        urllib.request.urlretrieve(url, destination, reporthook=_progress)
        print()

    print(f"\nDone. Assets are in ./{DATA_ROOT}/")


if __name__ == "__main__":
    main()
