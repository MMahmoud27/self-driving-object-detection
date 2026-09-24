"""Streamlit front end for the YOLOv3 detector.

Run it with:

    streamlit run app/app.py

All the detection logic lives in the ``roadvision`` package, so this file is
just the interface.
"""

import os
import sys
import tempfile

import streamlit as st
from PIL import Image

# Make the package importable when the app is launched from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from roadvision.yolo import COCO_LABELS, detect_image, load_darknet

DEFAULT_MODEL_PATH = os.environ.get("YOLO_WEIGHTS", "data/yolo_weights.h5")

st.set_page_config(page_title="Object Detection for Self-Driving Cars",
                   page_icon="🚗", layout="wide")


@st.cache_resource(show_spinner="Loading YOLOv3 weights...")
def get_model(model_path):
    """Load the network once and reuse it across reruns.

    Streamlit re-executes the whole script on every interaction, so without this
    cache the 237 MB model would be reloaded each time a slider moved.
    """
    return load_darknet(model_path)


st.title("🚗 Object Detection for Self-Driving Cars")
st.caption(
    "YOLOv3 running over the 80 COCO classes. Upload a road scene and the model "
    "marks every object it finds, with a confidence score."
)

with st.sidebar:
    st.header("Settings")

    model_path = st.text_input("Model weights", value=DEFAULT_MODEL_PATH)

    obj_thresh = st.slider(
        "Objectness threshold", 0.0, 1.0, 0.4, 0.05,
        help="How sure the model must be before a box counts as an object. "
             "Raise it to cut false positives, lower it to catch faint objects.",
    )
    nms_thresh = st.slider(
        "NMS (overlap) threshold", 0.0, 1.0, 0.45, 0.05,
        help="How much two boxes may overlap before the weaker one is dropped. "
             "Lower is more aggressive at removing duplicates.",
    )

    st.divider()
    st.subheader("Detectable classes")
    st.caption(", ".join(COCO_LABELS))

uploaded_file = st.file_uploader(
    "Upload a road scene", type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Upload an image to get started — a dashcam frame or street photo works best.")
    st.stop()

if not os.path.exists(model_path):
    st.error(
        f"Model weights not found at `{model_path}`.\n\n"
        "Download them first (see the README) or point the sidebar at the right path."
    )
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
darknet = get_model(model_path)

left, right = st.columns(2)
with left:
    st.subheader("Input")
    st.image(image, use_container_width=True)

with right:
    st.subheader("Detections")
    with st.spinner("Detecting..."):
        detected = detect_image(
            image, darknet, obj_thresh=obj_thresh, nms_thresh=nms_thresh
        )
    st.image(detected, use_container_width=True)

# Offer the annotated image as a download.
with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
    detected.save(handle.name)
    with open(handle.name, "rb") as fh:
        st.download_button(
            "Download annotated image",
            data=fh.read(),
            file_name=f"detected_{uploaded_file.name.rsplit('.', 1)[0]}.png",
            mime="image/png",
        )
os.unlink(handle.name)
