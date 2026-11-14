import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import av
import cv2
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
from ultralytics import YOLO

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
BASE = Path(__file__).resolve().parent
MODEL_FILE = BASE / "models" / "best.pt"
OUTPUT_DIR = BASE / "outputs"
HISTORY_FILE = OUTPUT_DIR / "detection_history.csv"
OUTPUT_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Smart Helmet AI | Safety Console",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

HISTORY_COLUMNS = [
    "timestamp", "source", "helmet_detections",
    "no_helmet_detections", "virtual_ignition", "status",
]

# ------------------------------------------------------------
# MODEL + DETECTION LOGIC
# ------------------------------------------------------------
@st.cache_resource
def load_model(model_path: str, modified: float):
    return YOLO(model_path)


def get_model():
    if not MODEL_FILE.exists():
        st.error("Model file not found: models/best.pt")
        st.info("Place your trained best.pt file inside the project's models folder.")
        st.stop()
    return load_model(str(MODEL_FILE), MODEL_FILE.stat().st_mtime)


def classify_name(name):
    value = str(name).lower().replace("_", " ").replace("-", " ").strip()
    if any(term in value for term in ("no helmet", "without helmet", "nohelmet", "withouthelmet")):
        return "no_helmet"
    if "helmet" in value:
        return "helmet"
    return "other"


def inspect_result(result, names):
    rows = []
    helmets = 0
    no_helmets = 0

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        label = names[class_id] if isinstance(names, (list, tuple)) else names.get(class_id, str(class_id))
        category = classify_name(label)

        if category == "helmet":
            helmets += 1
        elif category == "no_helmet":
            no_helmets += 1

        rows.append({
            "Detected class": str(label),
            "Category": category.replace("_", " ").title(),
            "Confidence": round(confidence * 100, 1),
        })

    if no_helmets:
        status, ignition = "NO HELMET DETECTED", "OFF"
    elif helmets:
        status, ignition = "HELMET DETECTED", "ON"
    else:
        status, ignition = "STATUS UNKNOWN", "OFF"

    return rows, helmets, no_helmets, status, ignition


# ------------------------------------------------------------
# HISTORY
# ------------------------------------------------------------
def read_history():
    if HISTORY_FILE.exists():
        try:
            df = pd.read_csv(HISTORY_FILE)
            for column in HISTORY_COLUMNS:
                if column not in df.columns:
                    df[column] = ""
            return df[HISTORY_COLUMNS]
        except Exception:
            return pd.DataFrame(columns=HISTORY_COLUMNS)
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def log_result(source, helmets, no_helmets, status, ignition):
    row = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "helmet_detections": int(helmets),
        "no_helmet_detections": int(no_helmets),
        "virtual_ignition": ignition,
        "status": status,
    }])
    existing = read_history()
    updated = pd.concat([existing, row], ignore_index=True)
    updated.to_csv(HISTORY_FILE, index=False)

    st.session_state.last_result = {
        "timestamp": row.iloc[0]["timestamp"],
        "source": source,
        "helmets": int(helmets),
        "no_helmets": int(no_helmets),
        "status": status,
        "ignition": ignition,
    }


def get_stats(df):
    scans = len(df)
    helmet_count = int(pd.to_numeric(df["helmet_detections"], errors="coerce").fillna(0).sum()) if scans else 0
    no_helmet_count = int(pd.to_numeric(df["no_helmet_detections"], errors="coerce").fillna(0).sum()) if scans else 0
    ignition_on = int((df["virtual_ignition"] == "ON").sum()) if scans else 0
    total_objects = helmet_count + no_helmet_count
    compliance = helmet_count / total_objects * 100 if total_objects else 0
    return {
        "scans": scans,
        "helmets": helmet_count,
        "no_helmets": no_helmet_count,
        "ignition_on": ignition_on,
        "ignition_off": scans - ignition_on,
        "compliance": compliance,
    }


# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------
def init_state():
    defaults = {
        "last_result": {
            "timestamp": "No scan yet",
            "source": "—",
            "helmets": 0,
            "no_helmets": 0,
            "status": "STATUS UNKNOWN",
            "ignition": "OFF",
        },
        "last_image_signature": None,
        "video_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ------------------------------------------------------------
# CUSTOM UI
# ------------------------------------------------------------
CSS = """
<style>
:root {
  --bg:#081426; --panel:#10223f; --panel2:#0c1b33;
  --border:#20395e; --text:#e6edf7; --muted:#9bb0ca;
  --blue:#38bdf8; --green:#22c55e; --red:#fb7185; --amber:#fbbf24;
}
[data-testid="stAppViewContainer"] {
  background: radial-gradient(ellipse at 10% 0%, #17345b 0%, #081426 48%);
  color:var(--text);
}
[data-testid="stHeader"] {background:transparent;}
[data-testid="stSidebar"] {
  background:linear-gradient(180deg,#0e2342 0%,#081426 100%);
  border-right:1px solid var(--border);
}
.block-container {max-width:1400px;padding-top:2rem;padding-bottom:3rem;}
h1,h2,h3,h4,p,label,span {color:var(--text);}
.brand {font-size:1.35rem;font-weight:850;color:#fff;margin-bottom:0;}
.brand-sub {font-size:.78rem;color:var(--muted);margin:0 0 1.2rem 0;}
.hero {
  padding:1.5rem 1.6rem;border-radius:20px;
  background:linear-gradient(120deg,#12345b,#10223f 65%,#0b1b32);
  border:1px solid #294b76;margin-bottom:1.2rem;
}
.hero-title {font-size:2rem;font-weight:850;color:#fff;margin:0;}
.hero-sub {color:#b6c8df;margin:.35rem 0 0 0;}
.section-title {font-size:1.08rem;font-weight:750;margin:.6rem 0 .8rem 0;}
.kpi {
  background:linear-gradient(155deg,#142b4c,#0d1d36);
  border:1px solid var(--border);border-radius:16px;
  padding:1rem 1.1rem;min-height:125px;
}
.kpi-label {font-size:.74rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);}
.kpi-value {font-size:2rem;font-weight:850;color:#fff;margin:.35rem 0 .1rem 0;}
.kpi-note {font-size:.78rem;color:var(--muted);}
.panel {
  background:linear-gradient(155deg,#122846,#0c1b33);
  border:1px solid var(--border);border-radius:16px;padding:1rem 1.2rem;
}
.pill {display:inline-block;border-radius:999px;padding:.25rem .7rem;font-size:.75rem;font-weight:750;}
div.stButton > button, div.stDownloadButton > button {
  border-radius:11px;border:1px solid #31547e;font-weight:700;min-height:2.6rem;
}
div.stButton > button[kind="primary"] {background:linear-gradient(90deg,#2563eb,#0891b2);border:0;}
[data-testid="stFileUploader"] section {
  background:#10223f;border:1px dashed #41638b;border-radius:14px;
}
[data-testid="stMetric"] {
  background:#10223f;border:1px solid var(--border);border-radius:13px;padding:.7rem;
}
[data-testid="stSidebar"] [role="radiogroup"] {gap:.35rem;}
[data-testid="stSidebar"] [role="radiogroup"] label {
  padding:.45rem .65rem;border-radius:10px;border:1px solid transparent;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
  background:#17365b;border-color:#31547e;
}
small,.muted {color:var(--muted);}
@media(max-width:760px) {
 .block-container{padding:1rem .8rem 2rem;}
 .hero-title{font-size:1.5rem;}
 .kpi-value{font-size:1.55rem;}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def hero(title, subtitle, icon="🏍️"):
    st.markdown(
        f'<div class="hero"><div class="hero-title">{icon} {title}</div>'
        f'<div class="hero-sub">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def metric_card(label, value, note, accent="#38bdf8"):
    st.markdown(
        f'<div class="kpi" style="border-top:3px solid {accent}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def show_decision(status, ignition):
    left, right = st.columns(2)
    with left:
        if status == "HELMET DETECTED":
            st.success("✅ HELMET DETECTED")
        elif status == "NO HELMET DETECTED":
            st.error("❌ NO HELMET DETECTED")
        else:
            st.warning("⚠️ STATUS UNKNOWN")
    with right:
        if ignition == "ON":
            st.success("🔑 VIRTUAL IGNITION: ON")
        else:
            st.error("🔒 VIRTUAL IGNITION: OFF")


def page_dashboard():
    df = read_history()
    stats = get_stats(df)
    last = st.session_state.last_result

    hero("Safety Command Centre", "A clear overview of helmet detection and simulated ignition decisions.", "🛡️")

    cols = st.columns(4)
    cards = [
        ("Total scans", stats["scans"], "Image, video and webcam", "#38bdf8"),
        ("Helmet detections", stats["helmets"], "Detected helmet objects", "#22c55e"),
        ("No-helmet detections", stats["no_helmets"], "Potential safety violations", "#fb7185"),
        ("Helmet detection share", f'{stats["compliance"]:.1f}%', "Helmet ÷ classified detections", "#a78bfa"),
    ]
    for col, (label, value, note, color) in zip(cols, cards):
        with col:
            metric_card(label, value, note, color)

    st.markdown('<div class="section-title">Latest system decision</div>', unsafe_allow_html=True)
    show_decision(last["status"], last["ignition"])
    st.caption(f'Last scan: {last["timestamp"]} · Source: {last["source"]}')

    if df.empty:
        st.info("No scans yet. Select Image Scan, Video Scan or Live Camera from the sidebar to begin.")
        return

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">Detection trend</div>', unsafe_allow_html=True)
        chart_df = df[["helmet_detections", "no_helmet_detections"]].tail(20).reset_index(drop=True)
        st.line_chart(chart_df)
    with right:
        st.markdown('<div class="section-title">Scans by input type</div>', unsafe_allow_html=True)
        st.bar_chart(df["source"].value_counts())
    st.markdown('<div class="section-title">Recent activity</div>', unsafe_allow_html=True)
    st.dataframe(df.tail(8).iloc[::-1], use_container_width=True, hide_index=True)


def page_image():
    hero("Image Scan", "Upload a clear rider image and inspect the model's detections.", "🖼️")
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="image_upload")
    if uploaded is None:
        st.info("Supported formats: JPG and PNG. For best results, use a clear, well-lit image.")
        return

    confidence = st.session_state.confidence
    image_size = st.session_state.image_size
    signature = (uploaded.name, uploaded.size, confidence, image_size)
    image = Image.open(uploaded).convert("RGB")

    with st.spinner("Analysing image with YOLO…"):
        model = get_model()
        result = model.predict(image, conf=confidence, imgsz=image_size, verbose=False)[0]
    rows, helmets, no_helmets, status, ignition = inspect_result(result, model.names)

    original_col, result_col = st.columns(2)
    with original_col:
        st.markdown("**Original image**")
        st.image(image, use_container_width=True)
    with result_col:
        st.markdown("**AI detection result**")
        st.image(cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB), use_container_width=True)

    show_decision(status, ignition)
    c1, c2, c3 = st.columns(3)
    c1.metric("Helmet objects", helmets)
    c2.metric("No-helmet objects", no_helmets)
    c3.metric("Other / unclassified", sum(row["Category"] == "Other" for row in rows))

    if rows:
        st.markdown("**Detection details**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.warning("The model did not detect a recognised helmet class in this image.")

    if st.session_state.last_image_signature != signature:
        log_result("image", helmets, no_helmets, status, ignition)
        st.session_state.last_image_signature = signature
        st.toast("Image scan saved to history.")


def page_video():
    hero("Video Scan", "Process a clip, preview the annotated output and download the result.", "🎬")
    uploaded = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv"], key="video_upload")
    stride = st.select_slider(
        "Processing speed",
        options=[1, 2, 3, 5],
        value=2,
        help="1 analyses every frame for more detail. 2, 3 or 5 can reduce processing time.",
    )
    if uploaded is None:
        st.info("Tip: begin with a short MP4 clip to test performance.")
        return

    if st.button("▶ Process video", type="primary", use_container_width=True):
        model = get_model()
        input_path = None
        output_path = None
        cap = None
        writer = None

        try:
            suffix = Path(uploaded.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=OUTPUT_DIR) as tmp:
                tmp.write(uploaded.getvalue())
                input_path = tmp.name
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                st.error("Could not open this video. Please try another MP4 file.")
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0 or fps > 120:
                fps = 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if width <= 0 or height <= 0:
                st.error("The video dimensions could not be read.")
                return

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=OUTPUT_DIR) as tmp_out:
                output_path = tmp_out.name

            writer = cv2.VideoWriter(
                output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not writer.isOpened():
                st.error("Could not create the annotated video on this system.")
                return

            progress = st.progress(0)
            preview = st.empty()
            status_area = st.empty()
            frame_number = 0
            helmet_total = 0
            no_helmet_total = 0
            last_status = "STATUS UNKNOWN"
            last_ignition = "OFF"
            last_annotated = None

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_number % stride == 0 or last_annotated is None:
                    result = model.predict(
                        frame,
                        conf=st.session_state.confidence,
                        imgsz=st.session_state.image_size,
                        verbose=False,
                    )[0]
                    _, h, nh, last_status, last_ignition = inspect_result(result, model.names)
                    helmet_total += h
                    no_helmet_total += nh
                    last_annotated = result.plot()

                writer.write(last_annotated)
                frame_number += 1

                if frame_number == 1 or frame_number % 10 == 0:
                    preview.image(
                        cv2.cvtColor(last_annotated, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True,
                    )
                    status_area.caption(
                        f"Frame {frame_number}/{total_frames or '?'} · {last_status} · Virtual ignition {last_ignition}"
                    )
                if total_frames > 0:
                    progress.progress(min(frame_number / total_frames, 1.0))

            writer.release()
            writer = None
            cap.release()
            cap = None

            with open(output_path, "rb") as file:
                video_bytes = file.read()

            st.session_state.video_result = {
                "bytes": video_bytes,
                "frames": frame_number,
                "helmets": helmet_total,
                "no_helmets": no_helmet_total,
                "status": last_status,
                "ignition": last_ignition,
            }
            log_result("video", helmet_total, no_helmet_total, last_status, last_ignition)
            st.success(f"Video processing completed: {frame_number} frames.")
        except Exception as exc:
            st.error(f"Video processing failed: {exc}")
        finally:
            if writer is not None:
                writer.release()
            if cap is not None:
                cap.release()
            for path in (input_path, output_path):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    video_result = st.session_state.video_result
    if video_result:
        st.markdown("**Latest processed video**")
        show_decision(video_result["status"], video_result["ignition"])
        a, b, c = st.columns(3)
        a.metric("Frames processed", video_result["frames"])
        b.metric("Helmet detections", video_result["helmets"])
        c.metric("No-helmet detections", video_result["no_helmets"])
        st.download_button(
            "⬇ Download annotated MP4",
            data=video_result["bytes"],
            file_name="smart_helmet_annotated.mp4",
            mime="video/mp4",
            use_container_width=True,
        )


def page_webcam():
    hero("Live Camera", "Run real-time helmet detection using your browser camera.", "📹")
    st.warning("Browser camera access requires permission and a secure connection. On Streamlit Cloud, use HTTPS.")
    if not MODEL_FILE.exists():
        st.info("Add models/best.pt to enable live detection.")
        return

    model = get_model()

    class HelmetVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.model = model
            self.status = "STATUS UNKNOWN"
            self.ignition = "OFF"
            self.helmets = 0
            self.no_helmets = 0
            self.last_update = time.time()

        def recv(self, frame):
            image = frame.to_ndarray(format="bgr24")
            # Smaller frame for better CPU responsiveness on laptop/CPU hosting.
            height, width = image.shape[:2]
            max_width = 640
            if width > max_width:
                new_height = int(height * max_width / width)
                image = cv2.resize(image, (max_width, new_height))
            prediction = self.model.predict(
                image,
                conf=st.session_state.get("confidence", 0.25),
                imgsz=st.session_state.get("image_size", 416),
                verbose=False,
            )[0]
            _, self.helmets, self.no_helmets, self.status, self.ignition = inspect_result(
                prediction, self.model.names
            )
            self.last_update = time.time()
            return av.VideoFrame.from_ndarray(prediction.plot(), format="bgr24")

    ctx = webrtc_streamer(
        key="smart-helmet-live-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=HelmetVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

    if ctx.state.playing:
        st.info("Camera is running. Detection is shown on the annotated live video above.")
        processor = ctx.video_processor
        if processor:
            st.caption(
                f"Latest processed frame: {processor.status} · "
                f"Helmet: {processor.helmets} · No helmet: {processor.no_helmets}"
            )
            if st.button("Save current webcam result to history"):
                log_result(
                    "webcam",
                    processor.helmets,
                    processor.no_helmets,
                    processor.status,
                    processor.ignition,
                )
                st.success("Current webcam result saved.")
    else:
        st.caption("Press START above and allow camera permission in your browser.")


def page_ignition():
    hero("Virtual Ignition", "A software-only representation of the system's decision.", "🔑")
    last = st.session_state.last_result
    is_on = last["ignition"] == "ON"
    color = "#22c55e" if is_on else "#fb7185"
    symbol = "🔓" if is_on else "🔒"
    label = "IGNITION ON" if is_on else "IGNITION OFF"

    st.markdown(
        f'<div class="panel" style="text-align:center;padding:2.2rem;border-top:4px solid {color}">'
        f'<div style="font-size:4rem">{symbol}</div>'
        f'<div style="font-size:2.5rem;font-weight:900;color:{color}">{label}</div>'
        f'<div class="muted">{last["status"]}</div>'
        f'<div class="muted">Last source: {last["source"]} · {last["timestamp"]}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Decision rules")
    a, b, c = st.columns(3)
    with a:
        st.error("**No helmet detected**  \nVirtual ignition OFF")
    with b:
        st.success("**Helmet detected only**  \nVirtual ignition ON")
    with c:
        st.warning("**No clear detection**  \nVirtual ignition OFF")
    st.caption("This is only a software simulation. It does not start, stop or control a real motorcycle.")


def page_history():
    hero("Detection History", "Review, filter and export scans saved on this app instance.", "🕘")
    df = read_history()
    if df.empty:
        st.info("No history available yet. Run an image, video or webcam scan.")
        return

    filters = st.columns(2)
    with filters[0]:
        sources = st.multiselect("Filter by source", sorted(df["source"].dropna().unique()), default=sorted(df["source"].dropna().unique()))
    with filters[1]:
        statuses = st.multiselect("Filter by status", sorted(df["status"].dropna().unique()), default=sorted(df["status"].dropna().unique()))

    view = df[df["source"].isin(sources) & df["status"].isin(statuses)]
    st.dataframe(view.tail(200).iloc[::-1], use_container_width=True, hide_index=True)
    st.download_button(
        "⬇ Download filtered CSV",
        view.to_csv(index=False).encode("utf-8"),
        "smart_helmet_history.csv",
        "text/csv",
    )
    if st.button("Clear all history", type="secondary"):
        HISTORY_FILE.unlink(missing_ok=True)
        st.session_state.last_result = {
            "timestamp": "No scan yet", "source": "—", "helmets": 0,
            "no_helmets": 0, "status": "STATUS UNKNOWN", "ignition": "OFF",
        }
        st.rerun()


def page_reports():
    hero("Reports & Analytics", "Create a downloadable summary of your detection activity.", "📄")
    df = read_history()
    if df.empty:
        st.info("Run at least one scan to generate a report.")
        return

    stats = get_stats(df)
    a, b, c = st.columns(3)
    with a:
        metric_card("Total scans", stats["scans"], "Recorded sessions")
    with b:
        metric_card("Helmet detections", stats["helmets"], "All recorded detections", "#22c55e")
    with c:
        metric_card("No-helmet detections", stats["no_helmets"], "All recorded detections", "#fb7185")

    report_html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Smart Helmet Report</title>
<style>
body{{font-family:Arial,sans-serif;margin:32px;color:#172033}}
h1{{color:#12345b}}.cards{{display:flex;gap:16px;flex-wrap:wrap}}
.card{{padding:16px;border:1px solid #ccd5e0;border-radius:10px;min-width:150px}}
table{{border-collapse:collapse;width:100%;margin-top:24px}}
td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}
@media print{{button{{display:none}}}}
</style></head><body>
<h1>Smart Helmet Detection Report</h1>
<p>Generated: {datetime.now().strftime("%d %b %Y, %H:%M")}</p>
<div class="cards">
<div class="card"><b>Total scans</b><h2>{stats["scans"]}</h2></div>
<div class="card"><b>Helmet detections</b><h2>{stats["helmets"]}</h2></div>
<div class="card"><b>No-helmet detections</b><h2>{stats["no_helmets"]}</h2></div>
<div class="card"><b>Detection share</b><h2>{stats["compliance"]:.1f}%</h2></div>
</div>
{df.to_html(index=False, escape=True)}
<p><small>Academic simulation only. Not connected to a real motorcycle.</small></p>
</body></html>"""

    left, right = st.columns(2)
    with left:
        st.download_button(
            "⬇ Download CSV report",
            df.to_csv(index=False).encode("utf-8"),
            "smart_helmet_report.csv",
            "text/csv",
            use_container_width=True,
        )
    with right:
        st.download_button(
            "⬇ Download printable HTML report",
            report_html.encode("utf-8"),
            "smart_helmet_report.html",
            "text/html",
            use_container_width=True,
        )
    st.caption("Open the HTML report in a browser and select Print → Save as PDF.")
    st.dataframe(df.tail(20).iloc[::-1], use_container_width=True, hide_index=True)


def page_about():
    hero("About the System", "Project overview, workflow and important limitations.", "ℹ️")
    st.markdown("""
    ### How it works
    1. The user provides an image, video or webcam stream.
    2. OpenCV reads and prepares image/video frames.
    3. The trained YOLO model detects the classes it learned.
    4. Python applies the project's virtual ignition decision rule.
    5. The result is displayed and recorded in the history file.

    ### Decision logic
    - Any recognised `no_helmet` detection → virtual ignition **OFF**.
    - A recognised `helmet` detection with no `no_helmet` detection → virtual ignition **ON**.
    - No recognised detection → virtual ignition **OFF**.

    ### Technology
    Python · Streamlit · Ultralytics YOLO · OpenCV · Pandas · streamlit-webrtc

    ### Important limitations
    This model detects helmet objects; it does not prove that a helmet is being worn or fastened.
    It does not reliably associate each helmet with a particular rider or distinguish a rider from a passenger.
    This is an academic software simulation only. Never connect it directly to a real motorcycle ignition.
    """)


# ------------------------------------------------------------
# MAIN NAVIGATION
# ------------------------------------------------------------
init_state()

with st.sidebar:
    st.markdown('<div class="brand">🏍️ SMART HELMET AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Rider Safety Monitoring Console</div>', unsafe_allow_html=True)
    selected_page = st.radio(
        "MAIN MENU",
        [
            "🏠 Dashboard",
            "🖼️ Image Scan",
            "🎬 Video Scan",
            "📹 Live Camera",
            "🔑 Virtual Ignition",
            "🕘 Detection History",
            "📄 Reports",
            "ℹ️ About",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**⚙️ Detection settings**")
    st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05, key="confidence")
    st.selectbox("Inference image size", [320, 416, 512, 640], index=1, key="image_size",
                 help="Smaller sizes are usually faster on CPU, but may reduce detection accuracy.")
    if MODEL_FILE.exists():
        st.success("YOLO model file found")
    else:
        st.warning("best.pt not found")
    st.caption("Academic simulation · No real bike connection")

PAGES = {
    "🏠 Dashboard": page_dashboard,
    "🖼️ Image Scan": page_image,
    "🎬 Video Scan": page_video,
    "📹 Live Camera": page_webcam,
    "🔑 Virtual Ignition": page_ignition,
    "🕘 Detection History": page_history,
    "📄 Reports": page_reports,
    "ℹ️ About": page_about,
}
PAGES[selected_page]()
