from pathlib import Path
from datetime import datetime
import csv

import cv2
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

BASE = Path(__file__).resolve().parent
MODEL_FILE = BASE / "models" / "best.pt"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

st.set_page_config(page_title="Smart Helmet Detection", page_icon="🏍️", layout="wide")
st.title("🏍️ Smart Helmet Detection & Virtual Ignition")
st.caption("Laptop-only computer vision simulation — no physical bike or ignition connection.")

@st.cache_resource
def load_model(model_path, modified):
    return YOLO(model_path)

def get_model():
    if not MODEL_FILE.exists():
        st.error("Missing models/best.pt. Follow MODEL_SETUP.md to obtain or train a helmet-specific YOLO model.")
        st.stop()
    return load_model(str(MODEL_FILE), MODEL_FILE.stat().st_mtime)

def classify_name(name):
    s = str(name).lower().replace("_", " ").replace("-", " ").strip()
    if any(x in s for x in ("no helmet", "without helmet", "nohelmet", "withouthelmet")):
        return "no_helmet"
    if "helmet" in s or "with helmet" in s:
        return "helmet"
    return "other"

def inspect(result, names):
    rows, helmets, no_helmets = [], 0, 0
    for box in result.boxes:
        cid = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label = names[cid] if isinstance(names, (list, tuple)) else names.get(cid, str(cid))
        kind = classify_name(label)
        helmets += kind == "helmet"
        no_helmets += kind == "no_helmet"
        rows.append({"class": str(label), "type": kind, "confidence": round(conf, 4)})
    if no_helmets:
        status, ignition = "NO HELMET DETECTED", "OFF"
    elif helmets:
        status, ignition = "HELMET DETECTED", "ON"
    else:
        status, ignition = "STATUS UNKNOWN", "OFF"
    return rows, helmets, no_helmets, status, ignition

def log_result(source, helmets, no_helmets, ignition):
    path = OUT / "detection_history.csv"
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "source", "helmet_detections", "no_helmet_detections", "virtual_ignition"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), source, helmets, no_helmets, ignition])

def status_cards(status, ignition):
    a, b = st.columns(2)
    with a:
        if status == "HELMET DETECTED":
            st.success("✅ " + status)
        elif status == "NO HELMET DETECTED":
            st.error("❌ " + status)
        else:
            st.warning("⚠️ " + status)
    with b:
        st.success("🔑 VIRTUAL IGNITION: ON") if ignition == "ON" else st.error("🔒 VIRTUAL IGNITION: OFF")

st.markdown("### Choose a test mode")
image_tab, video_tab, webcam_tab, history_tab, about_tab = st.tabs(
    ["Image", "Uploaded video", "Live webcam", "History", "About"]
)

with image_tab:
    uploaded = st.file_uploader("Upload a rider image", type=["jpg", "jpeg", "png"], key="img")
    if uploaded:
        model = get_model()
        image = Image.open(uploaded).convert("RGB")
        result = model.predict(image, conf=0.25, imgsz=640, verbose=False)[0]
        rows, h, nh, status, ignition = inspect(result, model.names)
        c1, c2 = st.columns(2)
        c1.image(image, caption="Original", use_container_width=True)
        c2.image(cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB), caption="AI detection", use_container_width=True)
        status_cards(status, ignition)
        st.metric("Helmet detections", h)
        st.metric("No-helmet detections", nh)
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        log_result("image", h, nh, ignition)

with video_tab:
    st.write("Upload a short clip. The app processes frames and creates an annotated video.")
    video = st.file_uploader("Choose video", type=["mp4", "avi", "mov", "mkv"], key="vid")
    if video and st.button("Process uploaded video", type="primary"):
        model = get_model()
        input_path = OUT / "input_video"
        input_path.write_bytes(video.getvalue())
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            st.error("Could not open this file. Try an MP4 video.")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            if fps <= 0 or fps > 120:
                fps = 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            output_path = OUT / "annotated_video.mp4"
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            bar, preview, message = st.progress(0), st.empty(), st.empty()
            frame_no = total_h = total_nh = 0
            last_status, last_ignition = "STATUS UNKNOWN", "OFF"
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                result = model.predict(frame, conf=0.25, imgsz=640, verbose=False)[0]
                rows, h, nh, last_status, last_ignition = inspect(result, model.names)
                total_h += h
                total_nh += nh
                annotated = result.plot()
                writer.write(annotated)
                frame_no += 1
                if frame_no == 1 or frame_no % 5 == 0:
                    preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                    message.info(f"Frame {frame_no}/{total or '?'} — {last_status}; virtual ignition {last_ignition}")
                if total:
                    bar.progress(min(frame_no / total, 1.0))
            cap.release()
            writer.release()
            status_cards(last_status, last_ignition)
            st.write(f"Processed {frame_no} frames. Total helmet detections across frames: {total_h}; no-helmet detections: {total_nh}.")
            with output_path.open("rb") as f:
                st.download_button("Download annotated video", f, "smart_helmet_annotated.mp4", "video/mp4")
            log_result("video", total_h, total_nh, last_ignition)

with webcam_tab:
    st.write("Click START and allow browser camera permission. Click STOP when finished.")
    if MODEL_FILE.exists():
        model = get_model()
        class LiveProcessor(VideoProcessorBase):
            def __init__(self):
                self.model = model
                self.status = "STATUS UNKNOWN"
                self.ignition = "OFF"
                self.h = self.nh = 0
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                result = self.model.predict(img, conf=0.25, imgsz=640, verbose=False)[0]
                _, self.h, self.nh, self.status, self.ignition = inspect(result, self.model.names)
                return av.VideoFrame.from_ndarray(result.plot(), format="bgr24")
        ctx = webrtc_streamer(
            key="helmet-webcam",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=LiveProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        if ctx.video_processor:
            status_cards(ctx.video_processor.status, ctx.video_processor.ignition)
            st.caption(f"Current frame — helmet: {ctx.video_processor.h}; no helmet: {ctx.video_processor.nh}")
    else:
        st.warning("Add models/best.pt first, then refresh this page.")

with history_tab:
    path = OUT / "detection_history.csv"
    if path.exists():
        df = pd.read_csv(path)
        st.dataframe(df.tail(100), use_container_width=True)
        st.download_button("Download CSV history", df.to_csv(index=False).encode(), "detection_history.csv", "text/csv")
        if st.button("Clear history"):
            path.unlink(missing_ok=True)
            st.rerun()
    else:
        st.info("No detection history yet. Run an image or video test.")

with about_tab:
    st.markdown("""
    **Workflow:** webcam/upload → OpenCV frames → trained YOLO helmet detector → Python decision → virtual ignition display.

    **Decision rule:** any no-helmet detection = OFF; helmet detection with no no-helmet detection = ON; no reliable detection = OFF.

    **Limitations:** This demo does not prove that a helmet is worn or fastened, and does not associate each helmet with a particular rider. A production system needs rider/helmet association, pillion handling, temporal validation, extensive testing and certified hardware. Never connect this simulation to a real motorcycle.
    """)
