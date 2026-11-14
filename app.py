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
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────
# 1. BASIC SETUP
# ──────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
MODEL_FILE = BASE / "models" / "best.pt"

st.set_page_config(
    page_title="Smart Helmet Detection",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

HISTORY_COLUMNS = [
    "timestamp", "source", "helmet_detections",
    "no_helmet_detections", "virtual_ignition", "status",
]

# ──────────────────────────────────────────────────────────────
# 2. YOUR ORIGINAL YOLO LOGIC (unchanged behaviour)
# ──────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────
# 3. SESSION STATE (history is kept per visitor, in memory)
# ──────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "history": [],
        "last": {"status": "STATUS UNKNOWN", "ignition": "OFF", "source": "-", "time": "-", "h": 0, "nh": 0},
        "last_image_key": None,
        "video_result": None,
        "wc_state": None,
        "wc_time": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def log_result(source, helmets, no_helmets, status, ignition):
    now = datetime.now().isoformat(timespec="seconds")
    st.session_state.history.append({
        "timestamp": now, "source": source,
        "helmet_detections": int(helmets), "no_helmet_detections": int(no_helmets),
        "virtual_ignition": ignition, "status": status,
    })
    st.session_state.last = {
        "status": status, "ignition": ignition, "source": source,
        "time": now, "h": int(helmets), "nh": int(no_helmets),
    }


def history_df():
    return pd.DataFrame(st.session_state.history, columns=HISTORY_COLUMNS)


def compute_stats(df):
    total = len(df)
    h = int(df["helmet_detections"].sum()) if total else 0
    nh = int(df["no_helmet_detections"].sum()) if total else 0
    on = int((df["virtual_ignition"] == "ON").sum()) if total else 0
    rate = (h / (h + nh) * 100) if (h + nh) else None
    return {"total": total, "h": h, "nh": nh, "on": on, "off": total - on, "rate": rate}


# ──────────────────────────────────────────────────────────────
# 4. THEME (dark navy CSS) + SMALL UI HELPERS
# ──────────────────────────────────────────────────────────────
CSS = """
<style>
:root{--bg:#0a1128;--panel:#111c3a;--panel2:#0d1733;--border:#1f2f5c;--text:#e6edff;--muted:#8fa3c7;--accent:#38bdf8;}
[data-testid="stAppViewContainer"]{background:radial-gradient(1200px 600px at 10% -10%,#14234d 0%,#0a1128 55%);color:var(--text);}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1b3d 0%,#081029 100%);border-right:1px solid var(--border);}
#MainMenu,footer{visibility:hidden;}
.block-container{padding-top:2rem;padding-bottom:3rem;max-width:1250px;}
h1,h2,h3,h4,p,label,span{color:var(--text);}
.page-title{font-size:1.9rem;font-weight:800;margin:0;color:#fff;}
.page-sub{color:var(--muted);margin:.15rem 0 1.2rem 0;}
.card{background:linear-gradient(160deg,var(--panel) 0%,var(--panel2) 100%);border:1px solid var(--border);border-radius:16px;padding:1rem 1.2rem;box-shadow:0 6px 24px rgba(0,0,0,.25);margin-bottom:.8rem;}
.kpi-label{color:var(--muted);font-size:.75rem;text-transform:uppercase;letter-spacing:.08em;}
.kpi-value{font-size:2rem;font-weight:800;line-height:1.25;color:#fff;}
.kpi-sub{color:var(--muted);font-size:.8rem;}
.status-text{font-size:1.15rem;font-weight:800;line-height:1.5;}
.ign-wrap{display:flex;align-items:center;gap:1.4rem;flex-wrap:wrap;}
.ign-ring{width:110px;height:110px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:2.4rem;border:4px solid;}
.ign-big{font-size:2.2rem;font-weight:900;line-height:1.1;}
.brand{font-size:1.25rem;font-weight:800;color:#fff;margin-bottom:.1rem;}
.brand-sub{color:var(--muted);font-size:.8rem;margin-bottom:1rem;}
.badge{display:inline-block;padding:.15rem .6rem;border-radius:999px;font-size:.75rem;font-weight:700;}
[data-testid="stSidebar"] div[role="radiogroup"]{gap:.25rem;}
[data-testid="stSidebar"] div[role="radiogroup"] label{border:1px solid transparent;border-radius:10px;padding:.5rem .7rem;width:100%;cursor:pointer;}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover{background:#13224a;border-color:var(--border);}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked){background:linear-gradient(90deg,#1d4ed8,#0ea5e9);border-color:#38bdf8;}
[data-testid="stMetric"]{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:.7rem 1rem;}
.stButton>button,.stDownloadButton>button{border-radius:10px;border:1px solid var(--border);font-weight:600;}
[data-testid="stFileUploader"] section{background:var(--panel);border:1px dashed #2c4380;border-radius:14px;}
@media (max-width:768px){
 .block-container{padding:1rem .8rem 2rem;}
 .page-title{font-size:1.45rem;}
 .kpi-value{font-size:1.6rem;}
 .ign-big{font-size:1.7rem;}
 .ign-ring{width:84px;height:84px;font-size:1.8rem;}
}
</style>
"""


def page_header(icon, title, sub):
    st.markdown(
        f'<div class="page-title">{icon} {title}</div><div class="page-sub">{sub}</div>',
        unsafe_allow_html=True,
    )


def kpi(label, value, sub="", color="#38bdf8"):
    return (
        f'<div class="card" style="border-left:4px solid {color}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>'
    )


def status_cards(status, ignition):
    color, icon = {"HELMET DETECTED": ("#22c55e", "✅"), "NO HELMET DETECTED": ("#ef4444", "❌")}.get(
        status, ("#f59e0b", "⚠️"))
    ign_color = "#22c55e" if ignition == "ON" else "#ef4444"
    ign_icon = "🔑" if ignition == "ON" else "🔒"
    a, b = st.columns(2)
    a.markdown(
        f'<div class="card" style="border-left:4px solid {color}"><div class="kpi-label">Helmet status</div>'
        f'<div class="status-text" style="color:{color}">{icon} {status}</div></div>',
        unsafe_allow_html=True)
    b.markdown(
        f'<div class="card" style="border-left:4px solid {ign_color}"><div class="kpi-label">Virtual ignition</div>'
        f'<div class="status-text" style="color:{ign_color}">{ign_icon} VIRTUAL IGNITION: {ignition}</div></div>',
        unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# 5. PAGES
# ──────────────────────────────────────────────────────────────
def page_dashboard():
    page_header("📊", "Dashboard", "Live overview of every detection run in this session.")
    df = history_df()
    s = compute_stats(df)
    last = st.session_state.last

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("Total scans", s["total"], "image / video / webcam runs", "#38bdf8"), unsafe_allow_html=True)
    c2.markdown(kpi("Helmet detections", s["h"], "all runs combined", "#22c55e"), unsafe_allow_html=True)
    c3.markdown(kpi("No-helmet detections", s["nh"], "all runs combined", "#ef4444"), unsafe_allow_html=True)
    rate = f"{s['rate']:.0f}%" if s["rate"] is not None else "—"
    c4.markdown(kpi("Helmet compliance", rate, "helmet ÷ (helmet + no-helmet)", "#a78bfa"), unsafe_allow_html=True)

    st.markdown("#### Current virtual ignition")
    status_cards(last["status"], last["ignition"])
    st.caption(f"Last result from: {last['source']} · {last['time']}")

    if s["total"] == 0:
        st.info("No detection history yet. Open **Image Detection**, **Video Detection** or **Live Webcam** to start.")
        return

    left, right = st.columns(2)
    with left:
        st.markdown("#### Helmet vs no-helmet per scan")
        st.line_chart(df[["helmet_detections", "no_helmet_detections"]].reset_index(drop=True),
                      color=["#22c55e", "#ef4444"])
    with right:
        st.markdown("#### Scans by source")
        st.bar_chart(df["source"].value_counts(), color="#38bdf8")

    st.markdown("#### Ignition decisions")
    d1, d2 = st.columns(2)
    d1.markdown(kpi("Ignition ON", s["on"], "helmet detected, no violations", "#22c55e"), unsafe_allow_html=True)
    d2.markdown(kpi("Ignition OFF", s["off"], "no helmet or unknown", "#ef4444"), unsafe_allow_html=True)


def page_image():
    page_header("🖼️", "Image Detection", "Upload a rider photo and the YOLO model checks for a helmet.")
    uploaded = st.file_uploader("Upload a rider image", type=["jpg", "jpeg", "png"], key="img")
    if not uploaded:
        st.info("Upload a JPG or PNG image to begin.")
        return
    model = get_model()
    conf, imgsz = st.session_state.conf, st.session_state.imgsz
    image = Image.open(uploaded).convert("RGB")
    with st.spinner("Running YOLO detection…"):
        result = model.predict(image, conf=conf, imgsz=imgsz, verbose=False)[0]
    rows, h, nh, status, ignition = inspect(result, model.names)

    c1, c2 = st.columns(2)
    c1.image(image, caption="Original")
    c2.image(cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB), caption="AI detection")
    status_cards(status, ignition)
    m1, m2 = st.columns(2)
    m1.metric("Helmet detections", h)
    m2.metric("No-helmet detections", nh)
    if rows:
        st.dataframe(pd.DataFrame(rows))

    # Log only once per uploaded file/settings (not on every page refresh)
    key = f"{uploaded.name}-{uploaded.size}-{conf}-{imgsz}"
    if st.session_state.last_image_key != key:
        log_result("image", h, nh, status, ignition)
        st.session_state.last_image_key = key


def page_video():
    page_header("🎬", "Video Detection", "Upload a short clip. Frames are analysed and an annotated MP4 is created.")
    video = st.file_uploader("Choose video", type=["mp4", "avi", "mov", "mkv"], key="vid")
    stride = st.select_slider(
        "Speed: analyse every N-th frame (1 = every frame, most accurate; higher = faster)",
        options=[1, 2, 3, 5], value=1)
    run = st.button("Process uploaded video", type="primary", disabled=video is None)

    if video and run:
        model = get_model()
        conf, imgsz = st.session_state.conf, st.session_state.imgsz
        suffix = Path(video.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(video.getvalue())
            in_path = tmp.name
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        cap = cv2.VideoCapture(in_path)
        try:
            if not cap.isOpened():
                st.error("Could not open this file. Try an MP4 video.")
            else:
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                if fps <= 0 or fps > 120:
                    fps = 25.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                bar, preview, message = st.progress(0), st.empty(), st.empty()
                frame_no = total_h = total_nh = frames_h = frames_nh = 0
                last_status, last_ignition = "STATUS UNKNOWN", "OFF"
                annotated = None
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_no % stride == 0 or annotated is None:
                        result = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
                        rows, h, nh, last_status, last_ignition = inspect(result, model.names)
                        total_h += h
                        total_nh += nh
                        frames_h += h > 0
                        frames_nh += nh > 0
                        annotated = result.plot()
                    writer.write(annotated)
                    frame_no += 1
                    if frame_no == 1 or frame_no % 5 == 0:
                        preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
                        message.info(f"Frame {frame_no}/{total or '?'} — {last_status}; virtual ignition {last_ignition}")
                    if total:
                        bar.progress(min(frame_no / total, 1.0))
                writer.release()
                with open(out_path, "rb") as f:
                    data = f.read()
                st.session_state.video_result = {
                    "bytes": data, "frames": frame_no, "h": total_h, "nh": total_nh,
                    "frames_h": frames_h, "frames_nh": frames_nh,
                    "status": last_status, "ignition": last_ignition,
                }
                log_result("video", total_h, total_nh, last_status, last_ignition)
                bar.empty(); preview.empty(); message.empty()
        finally:
            cap.release()
            for p in (in_path, out_path):
                try:
                    os.remove(p)
                except OSError:
                    pass

    r = st.session_state.video_result
    if r:
        st.success(f"Finished — processed {r['frames']} frames.")
        status_cards(r["status"], r["ignition"])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Frames", r["frames"])
        m2.metric("Helmet detections", r["h"])
        m3.metric("No-helmet detections", r["nh"])
        m4.metric("Frames with no-helmet", r["frames_nh"])
        st.download_button("⬇️ Download annotated video", r["bytes"], "smart_helmet_annotated.mp4", "video/mp4")
    elif not video:
        st.info("Choose a video file, then press **Process uploaded video**.")


def page_webcam():
    page_header("📹", "Live Webcam Detection", "Click START and allow browser camera permission. Click STOP when finished.")
    if not MODEL_FILE.exists():
        st.warning("Add models/best.pt first, then refresh this page.")
        return
    model = get_model()

    class LiveProcessor(VideoProcessorBase):
        def __init__(self):
            self.model = model
            self.status = "STATUS UNKNOWN"
            self.ignition = "OFF"
            self.h = self.nh = 0
            self.conf = 0.25
            self.imgsz = 640

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            result = self.model.predict(img, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
            _, self.h, self.nh, self.status, self.ignition = inspect(result, self.model.names)
            return av.VideoFrame.from_ndarray(result.plot(), format="bgr24")

    ctx = webrtc_streamer(
        key="helmet-webcam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=LiveProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    panel = st.empty()
    if not ctx.state.playing:
        with panel.container():
            st.info("Camera is stopped. Press START above to begin live detection.")
        return

    # Keep refreshing the status cards while the camera is running
    while ctx.state.playing:
        vp = ctx.video_processor
        if vp:
            vp.conf = float(st.session_state.conf)
            vp.imgsz = int(st.session_state.imgsz)
            with panel.container():
                status_cards(vp.status, vp.ignition)
                st.caption(f"Current frame — helmet: {vp.h}; no helmet: {vp.nh}")
            state = (vp.status, vp.ignition)
            now = time.time()
            if (vp.status != "STATUS UNKNOWN" and state != st.session_state.wc_state
                    and now - st.session_state.wc_time >= 3):
                log_result("webcam", vp.h, vp.nh, vp.status, vp.ignition)
                st.session_state.wc_state, st.session_state.wc_time = state, now
        time.sleep(0.5)


def page_ignition():
    page_header("🔑", "Virtual Ignition Status", "Simulated bike ignition — it never controls a real motorcycle.")
    last = st.session_state.last
    on = last["ignition"] == "ON"
    color = "#22c55e" if on else "#ef4444"
    st.markdown(
        f'<div class="card" style="border-left:4px solid {color}"><div class="ign-wrap">'
        f'<div class="ign-ring" style="border-color:{color};box-shadow:0 0 28px {color}55">{"🔑" if on else "🔒"}</div>'
        f'<div><div class="kpi-label">Ignition state</div>'
        f'<div class="ign-big" style="color:{color}">{"ON" if on else "OFF"}</div>'
        f'<div class="kpi-sub">{last["status"]} · source: {last["source"]} · {last["time"]}</div></div>'
        f'</div></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi("Rule 1", "No helmet → OFF", "any no-helmet detection locks ignition", "#ef4444"), unsafe_allow_html=True)
    c2.markdown(kpi("Rule 2", "Helmet → ON", "helmet found and no no-helmet found", "#22c55e"), unsafe_allow_html=True)
    c3.markdown(kpi("Rule 3", "Unknown → OFF", "no reliable detection keeps it locked", "#f59e0b"), unsafe_allow_html=True)

    df = history_df()
    st.markdown("#### Recent ignition events")
    if len(df):
        st.dataframe(df[["timestamp", "source", "status", "virtual_ignition"]].tail(10).iloc[::-1])
    else:
        st.info("No ignition events yet. Run a detection first.")
    if st.button("Reset ignition to OFF"):
        st.session_state.last = {"status": "STATUS UNKNOWN", "ignition": "OFF", "source": "-", "time": "-", "h": 0, "nh": 0}
        st.rerun()


def page_history():
    page_header("🕘", "Detection History", "Every image, video and webcam result from this session.")
    df = history_df()
    if df.empty:
        st.info("No detection history yet. Run an image, video or webcam test.")
        return
    sources = st.multiselect("Filter by source", sorted(df["source"].unique()), default=sorted(df["source"].unique()))
    view = df[df["source"].isin(sources)]
    st.dataframe(view.tail(100).iloc[::-1])
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download CSV history", df.to_csv(index=False).encode(), "detection_history.csv", "text/csv")
    if c2.button("🗑️ Clear history"):
        st.session_state.history = []
        st.rerun()


REPORT_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Smart Helmet Detection Report</title>
<style>body{font-family:Arial,sans-serif;background:#0a1128;color:#e6edff;margin:0;padding:32px;}
h1{margin:0 0 4px 0;}.sub{color:#8fa3c7;margin-bottom:24px;}
.grid{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:24px;}
.k{background:#111c3a;border:1px solid #1f2f5c;border-radius:12px;padding:14px 20px;min-width:150px;}
.k b{display:block;font-size:26px;color:#fff;}.k span{color:#8fa3c7;font-size:12px;text-transform:uppercase;}
table{border-collapse:collapse;width:100%;font-size:13px;}th,td{border-bottom:1px solid #1f2f5c;padding:8px;text-align:left;}
th{background:#111c3a;}.note{color:#8fa3c7;font-size:12px;margin-top:24px;}
@media print{body{background:#fff;color:#000}.k{background:#fff;border-color:#999}.k b{color:#000}th{background:#eee}}</style></head><body>
<h1>Smart Helmet Detection Report</h1><div class="sub">Generated __GEN__</div>
<div class="grid"><div class="k"><span>Total scans</span><b>__TOTAL__</b></div>
<div class="k"><span>Helmet detections</span><b>__H__</b></div>
<div class="k"><span>No-helmet detections</span><b>__NH__</b></div>
<div class="k"><span>Helmet compliance</span><b>__RATE__</b></div>
<div class="k"><span>Ignition ON / OFF</span><b>__ON__ / __OFF__</b></div></div>
<h3>Detection records</h3>__TABLE__
<div class="note">Simulated academic prototype. It does not control a real motorcycle.</div></body></html>"""


def build_report_html(df, s):
    table = df.to_html(index=False, border=0) if len(df) else "<p>No detection records.</p>"
    rate = f"{s['rate']:.0f}%" if s["rate"] is not None else "—"
    return (REPORT_TEMPLATE
            .replace("__GEN__", datetime.now().strftime("%d %b %Y, %H:%M"))
            .replace("__TOTAL__", str(s["total"])).replace("__H__", str(s["h"]))
            .replace("__NH__", str(s["nh"])).replace("__RATE__", rate)
            .replace("__ON__", str(s["on"])).replace("__OFF__", str(s["off"]))
            .replace("__TABLE__", table))


def page_reports():
    page_header("📄", "Reports", "Download your session results as CSV or as a printable HTML report.")
    df = history_df()
    s = compute_stats(df)
    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi("Total scans", s["total"], "", "#38bdf8"), unsafe_allow_html=True)
    c2.markdown(kpi("Helmet / No-helmet", f"{s['h']} / {s['nh']}", "total detections", "#22c55e"), unsafe_allow_html=True)
    rate = f"{s['rate']:.0f}%" if s["rate"] is not None else "—"
    c3.markdown(kpi("Compliance", rate, "helmet share of detections", "#a78bfa"), unsafe_allow_html=True)
    if df.empty:
        st.info("Nothing to report yet. Run a detection first.")
        return
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download CSV", df.to_csv(index=False).encode(), "detection_history.csv", "text/csv")
    d2.download_button("⬇️ Download HTML report", build_report_html(df, s).encode("utf-8"),
                       "smart_helmet_report.html", "text/html")
    st.caption("Tip: open the HTML report in your browser and choose Print → Save as PDF.")
    st.dataframe(df.tail(50).iloc[::-1])


def page_about():
    page_header("ℹ️", "About", "How the system works and what it can't do.")
    st.markdown("""
**Workflow:** webcam/upload → OpenCV frames → trained YOLO helmet detector → Python decision → virtual ignition display.

**Decision rule:** any no-helmet detection = OFF; helmet detection with no no-helmet detection = ON; no reliable detection = OFF.

**Tech stack:** Python · YOLO (Ultralytics) · OpenCV · Streamlit · streamlit-webrtc.

**Limitations:** This demo does not prove that a helmet is worn or fastened, and does not associate each helmet with a particular rider. A production system needs rider/helmet association, pillion handling, temporal validation, extensive testing and certified hardware. Never connect this simulation to a real motorcycle.
""")


# ──────────────────────────────────────────────────────────────
# 6. MAIN: sidebar navigation + page routing
# ──────────────────────────────────────────────────────────────
PAGES = {
    "📊 Dashboard": page_dashboard,
    "🖼️ Image Detection": page_image,
    "🎬 Video Detection": page_video,
    "📹 Live Webcam": page_webcam,
    "🔑 Ignition Status": page_ignition,
    "🕘 History": page_history,
    "📄 Reports": page_reports,
    "ℹ️ About": page_about,
}

st.markdown(CSS, unsafe_allow_html=True)
init_state()

with st.sidebar:
    st.markdown('<div class="brand">🏍️ Smart Helmet</div><div class="brand-sub">Detection &amp; Ignition Safety Simulation</div>',
                unsafe_allow_html=True)
    nav = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
    st.markdown("---")
    with st.expander("⚙️ Detection settings"):
        st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05, key="conf")
        st.selectbox("Image size (smaller = faster)", [640, 512, 416, 320], index=0, key="imgsz")
    if MODEL_FILE.exists():
        st.markdown('<span class="badge" style="background:#0f3d2a;color:#4ade80">● Model loaded: best.pt</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge" style="background:#4a1620;color:#f87171">● models/best.pt missing</span>', unsafe_allow_html=True)
    st.caption("Academic simulation only. Not connected to a real bike.")

PAGES[nav]()
