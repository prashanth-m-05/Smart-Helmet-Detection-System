import os
import tempfile
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
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
    "timestamp", "source", "helmet_detections", "no_helmet_detections",
    "virtual_ignition", "status", "avg_confidence",
]
ACC_COLUMNS = ["time", "file", "actual", "predicted", "helmets", "no_helmets"]

DEFAULT_CFG = {
    "conf": 0.25, "iou": 0.45, "imgsz": 640, "augment": False,
    "enhance": False, "smooth_n": 5, "style": "Custom colours",
}

THEMES = {  # name: (accent, accent_dark)
    "Navy Blue": ("#38bdf8", "#1d4ed8"),
    "Emerald Night": ("#34d399", "#047857"),
    "Violet Glow": ("#a78bfa", "#6d28d9"),
    "Sunset": ("#fb923c", "#c2410c"),
}

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
# 3. ADVANCED DETECTION HELPERS
# ──────────────────────────────────────────────────────────────
def settings():
    """Read the sidebar settings (main thread only)."""
    s = st.session_state
    return {
        "conf": float(s.conf), "iou": float(s.iou), "imgsz": int(s.imgsz),
        "augment": bool(s.augment), "enhance": bool(s.enhance),
        "smooth_n": int(s.smooth_n), "style": s.style,
    }


def enhance_image(img):
    """CLAHE contrast boost on the brightness channel - helps in dim light."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def detect(model, img_bgr, cfg):
    """Run YOLO with the chosen advanced options. Returns (image used, result)."""
    if cfg["enhance"]:
        img_bgr = enhance_image(img_bgr)
    result = model.predict(
        img_bgr, conf=cfg["conf"], iou=cfg["iou"], imgsz=cfg["imgsz"],
        augment=cfg["augment"], verbose=False,
    )[0]
    return img_bgr, result


def smooth_decision(window):
    """Majority vote over the last N frames so the ignition does not flicker.
    Safety-biased: a sustained no-helmet (>=30% of frames) locks the ignition."""
    n = len(window)
    if n == 0:
        return "STATUS UNKNOWN", "OFF"
    no_h = sum(s == "NO HELMET DETECTED" for s in window)
    yes_h = sum(s == "HELMET DETECTED" for s in window)
    if no_h / n >= 0.3:
        return "NO HELMET DETECTED", "OFF"
    if yes_h / n >= 0.5:
        return "HELMET DETECTED", "ON"
    return "STATUS UNKNOWN", "OFF"


FONT = cv2.FONT_HERSHEY_SIMPLEX
KIND_COLOR = {"helmet": (94, 197, 34), "no_helmet": (68, 68, 239)}  # BGR green / red


def annotate(img, result, names, ignition, style):
    """Draw boxes: either YOLO's default or colour-coded boxes + ignition banner."""
    if style == "YOLO default":
        return result.plot()
    out = img.copy()
    hgt, wid = out.shape[:2]
    t = max(2, int(min(hgt, wid) / 300))
    fs = 0.45 + t * 0.12
    for box in result.boxes:
        cid = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label = names[cid] if isinstance(names, (list, tuple)) else names.get(cid, str(cid))
        color = KIND_COLOR.get(classify_name(label), (11, 158, 245))
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, t)
        text = f"{label} {conf:.0%}"
        (tw, th), bl = cv2.getTextSize(text, FONT, fs, 1)
        ytop = max(y1 - th - bl - 4, 0)
        cv2.rectangle(out, (x1, ytop), (x1 + tw + 6, ytop + th + bl + 4), color, -1)
        cv2.putText(out, text, (x1 + 3, ytop + th + 1), FONT, fs, (255, 255, 255), 1, cv2.LINE_AA)
    banner = f"IGNITION: {ignition}"
    bcol = (94, 197, 34) if ignition == "ON" else (68, 68, 239)
    (tw, th), bl = cv2.getTextSize(banner, FONT, fs + 0.2, 2)
    cv2.rectangle(out, (0, 0), (tw + 20, th + bl + 16), bcol, -1)
    cv2.putText(out, banner, (10, th + 8), FONT, fs + 0.2, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def avg_conf_of(rows):
    return float(np.mean([r["confidence"] for r in rows])) if rows else None


# Browser-side sound: a buzzer (Web Audio) and a spoken message (speech synthesis).
BUZZER_JS = """
try{
  var C=window.AudioContext||window.webkitAudioContext; var ctx=new C();
  if(ctx.state==='suspended'){ctx.resume();}
  var t=ctx.currentTime;
  for(var i=0;i<4;i++){
    var o=ctx.createOscillator(), g=ctx.createGain();
    o.type='square'; o.frequency.value=(i%2?740:880);
    o.connect(g); g.connect(ctx.destination);
    var s=t+i*0.28;
    g.gain.setValueAtTime(0.0001,s);
    g.gain.exponentialRampToValueAtTime(0.35,s+0.02);
    g.gain.exponentialRampToValueAtTime(0.0001,s+0.24);
    o.start(s); o.stop(s+0.25);
  }
}catch(e){}
"""
SPEECH_JS = """
try{
  var u=new SpeechSynthesisUtterance('Helmet detected');
  u.rate=1; u.pitch=1; u.volume=1;
  window.speechSynthesis.cancel(); window.speechSynthesis.speak(u);
}catch(e){}
"""


def play_alert(kind):
    """kind = 'buzzer' (no helmet) or 'speech' (says 'Helmet detected')."""
    if not st.session_state.get("sound", True):
        return
    js = BUZZER_JS if kind == "buzzer" else SPEECH_JS
    # time.time() in a comment makes every call unique so the browser plays it again
    components.html(f"<script>/*{time.time()}*/{js}</script>", height=0)


# ──────────────────────────────────────────────────────────────
# 4. SESSION STATE (kept per visitor, in memory)
# ──────────────────────────────────────────────────────────────
EMPTY_LAST = {"status": "STATUS UNKNOWN", "ignition": "OFF", "source": "-", "time": "-", "h": 0, "nh": 0}


def init_state():
    defaults = {
        "history": [], "last": dict(EMPTY_LAST), "logged_keys": [],
        "video_result": None, "wc_state": None, "wc_time": 0.0,
        "acc_records": [], "acc_keys": [],
        "wc_buzz_time": 0.0, "wc_spoken": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def log_result(source, helmets, no_helmets, status, ignition, avg_conf=None):
    now = datetime.now().isoformat(timespec="seconds")
    st.session_state.history.append({
        "timestamp": now, "source": source,
        "helmet_detections": int(helmets), "no_helmet_detections": int(no_helmets),
        "virtual_ignition": ignition, "status": status,
        "avg_confidence": round(avg_conf, 4) if avg_conf is not None else None,
    })
    st.session_state.last = {
        "status": status, "ignition": ignition, "source": source,
        "time": now, "h": int(helmets), "nh": int(no_helmets),
    }


def history_df():
    return pd.DataFrame(st.session_state.history, columns=HISTORY_COLUMNS)


def acc_df():
    return pd.DataFrame(st.session_state.acc_records, columns=ACC_COLUMNS)


def compute_stats(df):
    total = len(df)
    h = int(df["helmet_detections"].sum()) if total else 0
    nh = int(df["no_helmet_detections"].sum()) if total else 0
    on = int((df["virtual_ignition"] == "ON").sum()) if total else 0
    rate = (h / (h + nh) * 100) if (h + nh) else None
    conf = pd.to_numeric(df["avg_confidence"], errors="coerce").dropna() if total else []
    avg_conf = float(np.mean(conf)) if len(conf) else None
    return {"total": total, "h": h, "nh": nh, "on": on, "off": total - on, "rate": rate, "avg_conf": avg_conf}


def accuracy_stats(df):
    if df.empty:
        return None
    n = len(df)
    correct = int((df["actual"] == df["predicted"]).sum())
    tp = int(((df["actual"] == "No helmet") & (df["predicted"] == "No helmet")).sum())
    fp = int(((df["actual"] == "Helmet") & (df["predicted"] == "No helmet")).sum())
    fn = int(((df["actual"] == "No helmet") & (df["predicted"] != "No helmet")).sum())
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = 2 * prec * rec / (prec + rec) if (prec and rec) else None
    return {"n": n, "acc": correct / n, "prec": prec, "rec": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def pct(x):
    return f"{x * 100:.0f}%" if x is not None else "—"


# ──────────────────────────────────────────────────────────────
# 5. THEME (dark navy CSS) + SMALL UI HELPERS
# ──────────────────────────────────────────────────────────────
CSS = """
<style>
:root{--bg:#0a1128;--panel:#111c3a;--panel2:#0d1733;--border:#1f2f5c;--text:#e6edff;--muted:#8fa3c7;--accent:__A__;}
[data-testid="stAppViewContainer"]{background:radial-gradient(1200px 600px at 10% -10%,#14234d 0%,#0a1128 55%);color:var(--text);}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1b3d 0%,#081029 100%);border-right:1px solid var(--border);}
#MainMenu,footer{visibility:hidden;}
.block-container{padding-top:2rem;padding-bottom:3rem;max-width:1250px;}
h1,h2,h3,h4,p,label,span{color:var(--text);}
.page-title{font-size:1.9rem;font-weight:800;margin:0;color:#fff;}
.page-sub{color:var(--muted);margin:.15rem 0 1.2rem 0;}
.card{background:linear-gradient(160deg,var(--panel) 0%,var(--panel2) 100%);border:1px solid var(--border);border-radius:16px;padding:1rem 1.2rem;box-shadow:0 6px 24px rgba(0,0,0,.25);margin-bottom:.8rem;transition:transform .15s ease,box-shadow .15s ease;}
.card:hover{transform:translateY(-2px);box-shadow:0 10px 28px rgba(0,0,0,.35);}
.kpi-label{color:var(--muted);font-size:.75rem;text-transform:uppercase;letter-spacing:.08em;}
.kpi-value{font-size:2rem;font-weight:800;line-height:1.25;color:#fff;}
.kpi-sub{color:var(--muted);font-size:.8rem;}
.status-text{font-size:1.15rem;font-weight:800;line-height:1.5;}
.ign-wrap{display:flex;align-items:center;gap:1.4rem;flex-wrap:wrap;}
.ign-ring{width:110px;height:110px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:2.4rem;border:4px solid;animation:pulse 2.2s ease-in-out infinite;}
@keyframes pulse{0%,100%{transform:scale(1)}50%{transform:scale(1.07)}}
.ign-big{font-size:2.2rem;font-weight:900;line-height:1.1;}
.brand{font-size:1.25rem;font-weight:800;color:#fff;margin-bottom:.1rem;}
.brand-sub{color:var(--muted);font-size:.8rem;margin-bottom:1rem;}
.badge{display:inline-block;padding:.15rem .6rem;border-radius:999px;font-size:.75rem;font-weight:700;}
[data-testid="stSidebar"] div[role="radiogroup"]{gap:.25rem;}
[data-testid="stSidebar"] div[role="radiogroup"] label{border:1px solid transparent;border-radius:10px;padding:.5rem .7rem;width:100%;cursor:pointer;}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover{background:#13224a;border-color:var(--border);}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked){background:linear-gradient(90deg,__B__,__A__);border-color:__A__;}
[data-testid="stMetric"]{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:.7rem 1rem;}
.stButton>button,.stDownloadButton>button{border-radius:10px;border:1px solid var(--border);font-weight:600;}
.stButton>button[kind="primary"],[data-testid="stBaseButton-primary"]{background:linear-gradient(90deg,__B__,__A__);border:none;color:#fff;}
[data-testid="stFileUploader"] section{background:var(--panel);border:1px dashed #2c4380;border-radius:14px;}
[data-testid="stExpander"]{border:1px solid var(--border);border-radius:14px;background:var(--panel2);}
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
    st.markdown(f'<div class="page-title">{icon} {title}</div><div class="page-sub">{sub}</div>',
                unsafe_allow_html=True)


def kpi(label, value, sub="", color="#38bdf8"):
    return (f'<div class="card" style="border-left:4px solid {color}">'
            f'<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{sub}</div></div>')


def status_cards(status, ignition):
    color, icon = {"HELMET DETECTED": ("#22c55e", "✅"), "NO HELMET DETECTED": ("#ef4444", "❌")}.get(
        status, ("#f59e0b", "⚠️"))
    ign_color = "#22c55e" if ignition == "ON" else "#ef4444"
    ign_icon = "🔑" if ignition == "ON" else "🔒"
    a, b = st.columns(2)
    a.markdown(
        f'<div class="card" style="border-left:4px solid {color}"><div class="kpi-label">Helmet status</div>'
        f'<div class="status-text" style="color:{color}">{icon} {status}</div></div>', unsafe_allow_html=True)
    b.markdown(
        f'<div class="card" style="border-left:4px solid {ign_color}"><div class="kpi-label">Virtual ignition</div>'
        f'<div class="status-text" style="color:{ign_color}">{ign_icon} VIRTUAL IGNITION: {ignition}</div></div>',
        unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# 6. PAGES
# ──────────────────────────────────────────────────────────────
def page_dashboard():
    page_header("📊", "Dashboard", "Live overview of every detection run in this session.")
    df = history_df()
    s = compute_stats(df)
    last = st.session_state.last
    ast = accuracy_stats(acc_df())

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("Total scans", s["total"], "image / video / webcam runs"), unsafe_allow_html=True)
    c2.markdown(kpi("Helmet detections", s["h"], "all runs combined", "#22c55e"), unsafe_allow_html=True)
    c3.markdown(kpi("No-helmet detections", s["nh"], "all runs combined", "#ef4444"), unsafe_allow_html=True)
    c4.markdown(kpi("Helmet compliance", pct(s["rate"] / 100 if s["rate"] is not None else None),
                    "helmet ÷ (helmet + no-helmet)", "#a78bfa"), unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)
    e1.markdown(kpi("Avg. confidence", pct(s["avg_conf"]), "how sure the model is", "#38bdf8"), unsafe_allow_html=True)
    e2.markdown(kpi("Ignition ON rate", pct(s["on"] / s["total"]) if s["total"] else "—",
                    "share of scans allowed to start", "#22c55e"), unsafe_allow_html=True)
    e3.markdown(kpi("Accuracy test", pct(ast["acc"]) if ast else "—",
                    f"{ast['n']} labelled images" if ast else "add labels on Image page", "#f59e0b"),
                unsafe_allow_html=True)

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


def add_acc_record(key, name, actual, h, nh):
    predicted = "No helmet" if nh else ("Helmet" if h else "Unknown")
    st.session_state.acc_records.append({
        "time": datetime.now().isoformat(timespec="seconds"), "file": name,
        "actual": actual, "predicted": predicted, "helmets": int(h), "no_helmets": int(nh),
    })
    st.session_state.acc_keys.append(key)


def analyse_one(model, f, cfg, idx):
    image = Image.open(f).convert("RGB")
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    with st.spinner(f"Running YOLO on {f.name}…"):
        base, result = detect(model, bgr, cfg)
    rows, h, nh, status, ignition = inspect(result, model.names)
    shown = annotate(base, result, model.names, ignition, cfg["style"])
    ac = avg_conf_of(rows)

    key = f"{f.name}-{f.size}-{cfg['conf']}-{cfg['iou']}-{cfg['imgsz']}-{cfg['augment']}-{cfg['enhance']}"
    sound = None
    if key not in st.session_state.logged_keys:
        log_result("image", h, nh, status, ignition, ac)
        st.session_state.logged_keys.append(key)
        if nh:
            st.toast("No helmet detected — virtual ignition locked", icon="🚨")
            sound = "buzzer"
        elif h:
            st.toast("Helmet detected — virtual ignition ON", icon="✅")
            sound = "speech"

    icon = {"HELMET DETECTED": "✅", "NO HELMET DETECTED": "🚨"}.get(status, "⚠️")
    with st.expander(f"{icon} {f.name} — {status.title()}", expanded=(idx == 0)):
        c1, c2 = st.columns(2)
        c1.image(image, caption="Original")
        c2.image(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB), caption="AI detection")
        status_cards(status, ignition)
        m1, m2, m3 = st.columns(3)
        m1.metric("Helmet detections", h)
        m2.metric("No-helmet detections", nh)
        m3.metric("Avg. confidence", pct(ac))
        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                column_config={"confidence": st.column_config.ProgressColumn(
                    "confidence", min_value=0.0, max_value=1.0, format="%.2f")},
            )
        st.markdown("**Accuracy test (optional)** — tell the app what is really in this picture:")
        g1, g2 = st.columns([3, 1])
        gt = g1.radio("Actual answer", ["Skip", "Helmet", "No helmet"], horizontal=True,
                      key=f"gt_{key}", label_visibility="collapsed")
        if key in st.session_state.acc_keys:
            g2.success("Recorded")
        elif gt != "Skip" and g2.button("➕ Add", key=f"add_{key}"):
            add_acc_record(key, f.name, gt, h, nh)
            st.rerun()
    return sound


def page_image():
    page_header("🖼️", "Image Detection", "Upload one or many rider photos, or take a snapshot with your camera.")
    tab_up, tab_cam = st.tabs(["📁 Upload images", "📸 Camera snapshot"])
    files = []
    with tab_up:
        up = st.file_uploader("Upload rider images", type=["jpg", "jpeg", "png"],
                              accept_multiple_files=True, key="img")
        files += list(up or [])
    with tab_cam:
        snap = st.camera_input("Take a photo of the rider")
        if snap:
            files.append(snap)
    if not files:
        st.info("Upload JPG/PNG images or take a snapshot to begin.")
        return
    model = get_model()
    cfg = settings()
    sounds = [analyse_one(model, f, cfg, i) for i, f in enumerate(files)]
    if "buzzer" in sounds:        # any rider without a helmet -> buzzer
        play_alert("buzzer")
    elif "speech" in sounds:      # all helmets -> say "Helmet detected"
        play_alert("speech")


def page_video():
    page_header("🎬", "Video Detection", "Upload a short clip. Frames are analysed and an annotated MP4 is created.")
    video = st.file_uploader("Choose video", type=["mp4", "avi", "mov", "mkv"], key="vid")
    stride = st.select_slider(
        "Speed: analyse every N-th frame (1 = every frame, most accurate; higher = faster)",
        options=[1, 2, 3, 5], value=1)
    run = st.button("Process uploaded video", type="primary", disabled=video is None)

    if video and run:
        model = get_model()
        cfg = settings()
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
                frame_no = total_h = total_nh = frames_nh = conf_n = 0
                conf_sum = 0.0
                window = deque(maxlen=cfg["smooth_n"])
                timeline = []
                last_status, last_ignition = "STATUS UNKNOWN", "OFF"
                annotated = None
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_no % stride == 0 or annotated is None:
                        base, result = detect(model, frame, cfg)
                        rows, h, nh, raw_status, _ = inspect(result, model.names)
                        window.append(raw_status)
                        last_status, last_ignition = smooth_decision(window)
                        total_h += h
                        total_nh += nh
                        frames_nh += nh > 0
                        conf_sum += sum(r["confidence"] for r in rows)
                        conf_n += len(rows)
                        annotated = annotate(base, result, model.names, last_ignition, cfg["style"])
                        timeline.append({"frame": frame_no, "helmet": h, "no_helmet": nh,
                                         "ignition_on": int(last_ignition == "ON")})
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
                avg_conf = conf_sum / conf_n if conf_n else None
                st.session_state.video_result = {
                    "bytes": data, "frames": frame_no, "h": total_h, "nh": total_nh,
                    "frames_nh": frames_nh, "status": last_status, "ignition": last_ignition,
                    "avg_conf": avg_conf, "timeline": timeline,
                }
                log_result("video", total_h, total_nh, last_status, last_ignition, avg_conf)
                bar.empty(); preview.empty(); message.empty()
                if last_status == "NO HELMET DETECTED":
                    play_alert("buzzer")
                elif last_status == "HELMET DETECTED":
                    play_alert("speech")
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
        tl = pd.DataFrame(r["timeline"])
        on_pct = tl["ignition_on"].mean() if len(tl) else None
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Frames", r["frames"])
        m2.metric("Helmet / No-helmet", f"{r['h']} / {r['nh']}")
        m3.metric("Ignition ON time", pct(on_pct))
        m4.metric("Avg. confidence", pct(r["avg_conf"]))
        if len(tl):
            st.markdown("#### Detections over time")
            st.line_chart(tl.set_index("frame")[["helmet", "no_helmet"]], color=["#22c55e", "#ef4444"])
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
            self.cfg = dict(DEFAULT_CFG)
            self.window = deque(maxlen=DEFAULT_CFG["smooth_n"])
            self.status = "STATUS UNKNOWN"
            self.ignition = "OFF"
            self.h = self.nh = 0

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            cfg = self.cfg
            if self.window.maxlen != cfg["smooth_n"]:
                self.window = deque(self.window, maxlen=cfg["smooth_n"])
            base, result = detect(self.model, img, cfg)
            _, self.h, self.nh, raw, _ = inspect(result, self.model.names)
            self.window.append(raw)
            self.status, self.ignition = smooth_decision(self.window)
            out = annotate(base, result, self.model.names, self.ignition, cfg["style"])
            return av.VideoFrame.from_ndarray(out, format="bgr24")

    ctx = webrtc_streamer(
        key="helmet-webcam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=LiveProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.caption("Tip: turn off Test-time augmentation for smoother live video — it is slower.")

    panel = st.empty()
    sound_slot = st.empty()
    if not ctx.state.playing:
        with panel.container():
            st.info("Camera is stopped. Press START above to begin live detection.")
        return

    st.session_state.wc_spoken = None
    while ctx.state.playing:
        vp = ctx.video_processor
        if vp:
            vp.cfg = settings()
            with panel.container():
                status_cards(vp.status, vp.ignition)
                st.caption(f"Current frame — helmet: {vp.h}; no helmet: {vp.nh}")
            state = (vp.status, vp.ignition)
            now = time.time()
            if (vp.status != "STATUS UNKNOWN" and state != st.session_state.wc_state
                    and now - st.session_state.wc_time >= 3):
                log_result("webcam", vp.h, vp.nh, vp.status, vp.ignition)
                st.session_state.wc_state, st.session_state.wc_time = state, now
            # Sound: buzzer repeats every 3 s while there is no helmet;
            # "Helmet detected" is spoken once each time a helmet appears.
            if vp.status == "NO HELMET DETECTED":
                st.session_state.wc_spoken = None
                if now - st.session_state.wc_buzz_time >= 3:
                    with sound_slot:
                        play_alert("buzzer")
                    st.session_state.wc_buzz_time = now
            elif vp.status == "HELMET DETECTED" and st.session_state.wc_spoken != "HELMET DETECTED":
                with sound_slot:
                    play_alert("speech")
                st.session_state.wc_spoken = "HELMET DETECTED"
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
        st.session_state.last = dict(EMPTY_LAST)
        st.rerun()


def page_accuracy():
    page_header("🎯", "Accuracy & Model Info", "Measure how well the model works on YOUR test images.")
    tab_live, tab_train, tab_info = st.tabs(["✅ Live accuracy test", "📈 Training results", "🧠 Model info"])

    with tab_live:
        st.write("On the **Image Detection** page, choose the real answer for each picture and press **Add**. "
                 "The numbers below are calculated from those labelled pictures.")
        df = acc_df()
        a = accuracy_stats(df)
        if a is None:
            st.info("No labelled images yet. Use at least 30 varied images (day, night, different angles) for meaningful results.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(kpi("Accuracy", pct(a["acc"]), f"{a['n']} labelled images", "#22c55e"), unsafe_allow_html=True)
            c2.markdown(kpi("Precision (no-helmet)", pct(a["prec"]), "alarms that were real", "#38bdf8"), unsafe_allow_html=True)
            c3.markdown(kpi("Recall (no-helmet)", pct(a["rec"]), "violations caught", "#f59e0b"), unsafe_allow_html=True)
            c4.markdown(kpi("F1 score", pct(a["f1"]), "balance of the two", "#a78bfa"), unsafe_allow_html=True)
            st.markdown("#### Confusion matrix (rows = actual, columns = predicted)")
            cm = pd.crosstab(df["actual"], df["predicted"]).reindex(
                index=["Helmet", "No helmet"], columns=["Helmet", "No helmet", "Unknown"], fill_value=0)
            st.dataframe(cm)
            st.caption(f"Missed violations (no-helmet predicted as something else): {a['fn']} · "
                       f"False alarms (helmet predicted as no-helmet): {a['fp']}. "
                       "Recall matters most for safety. Low recall means violations are being missed.")
            if a["n"] < 30:
                st.warning("Fewer than 30 images — treat these numbers as rough only.")
            st.dataframe(df.iloc[::-1])
            d1, d2 = st.columns(2)
            d1.download_button("⬇️ Download accuracy records", df.to_csv(index=False).encode(),
                               "accuracy_records.csv", "text/csv")
            if d2.button("🗑️ Clear accuracy records"):
                st.session_state.acc_records, st.session_state.acc_keys = [], []
                st.rerun()

    with tab_train:
        models_dir = BASE / "models"
        csv_path = models_dir / "results.csv"
        shown = False
        if csv_path.exists():
            r = pd.read_csv(csv_path)
            r.columns = [c.strip() for c in r.columns]
            cols = {"Precision": "metrics/precision(B)", "Recall": "metrics/recall(B)",
                    "mAP@50": "metrics/mAP50(B)", "mAP@50-95": "metrics/mAP50-95(B)"}
            if all(c in r.columns for c in cols.values()):
                best = r.loc[r["metrics/mAP50-95(B)"].idxmax()]
                k = st.columns(4)
                for col, (label, name) in zip(k, cols.items()):
                    col.metric(label, pct(float(best[name])))
                st.caption("Values from the best training epoch.")
                if "epoch" in r.columns:
                    st.line_chart(r.set_index("epoch")[list(cols.values())])
                shown = True
        for fname in ["results.png", "confusion_matrix.png", "confusion_matrix_normalized.png",
                      "PR_curve.png", "F1_curve.png"]:
            p = models_dir / fname
            if p.exists():
                st.image(str(p), caption=fname)
                shown = True
        if not shown:
            st.info("No training files found. After training, copy `results.csv`, `results.png`, "
                    "`confusion_matrix.png` and `PR_curve.png` from your `runs/detect/train/` folder "
                    "into the `models/` folder of this project, then push to GitHub. They will appear here.")

    with tab_info:
        if MODEL_FILE.exists():
            model = get_model()
            names = model.names
            st.markdown("**Classes the model detects:**")
            st.dataframe(pd.DataFrame({
                "id": list(names.keys()) if isinstance(names, dict) else list(range(len(names))),
                "class": list(names.values()) if isinstance(names, dict) else list(names),
            }))
            st.markdown("**How each class is treated:** a name containing “no helmet” locks the ignition; "
                        "a name containing “helmet” unlocks it; anything else is ignored.")
        else:
            st.warning("models/best.pt not found.")


def page_history():
    page_header("🕘", "Detection History", "Every image, video and webcam result from this session.")
    df = history_df()
    if df.empty:
        st.info("No detection history yet. Run an image, video or webcam test.")
        return
    opts = sorted(df["source"].unique())
    sources = st.multiselect("Filter by source", opts, default=opts)
    st.dataframe(df[df["source"].isin(sources)].tail(100).iloc[::-1])
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download CSV history", df.to_csv(index=False).encode(), "detection_history.csv", "text/csv")
    if c2.button("🗑️ Clear history"):
        st.session_state.history = []
        st.session_state.logged_keys = []
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
<div class="k"><span>Avg. confidence</span><b>__CONF__</b></div>
<div class="k"><span>Ignition ON / OFF</span><b>__ON__ / __OFF__</b></div>
<div class="k"><span>Labelled accuracy</span><b>__ACC__</b></div></div>
<h3>Detection records</h3>__TABLE__
<div class="note">Simulated academic prototype. It does not control a real motorcycle.</div></body></html>"""


def build_report_html(df, s, a):
    table = df.to_html(index=False, border=0) if len(df) else "<p>No detection records.</p>"
    return (REPORT_TEMPLATE
            .replace("__GEN__", datetime.now().strftime("%d %b %Y, %H:%M"))
            .replace("__TOTAL__", str(s["total"])).replace("__H__", str(s["h"]))
            .replace("__NH__", str(s["nh"]))
            .replace("__RATE__", pct(s["rate"] / 100 if s["rate"] is not None else None))
            .replace("__CONF__", pct(s["avg_conf"]))
            .replace("__ON__", str(s["on"])).replace("__OFF__", str(s["off"]))
            .replace("__ACC__", pct(a["acc"]) if a else "—")
            .replace("__TABLE__", table))


def page_reports():
    page_header("📄", "Reports", "Download your session results as CSV or as a printable HTML report.")
    df = history_df()
    s = compute_stats(df)
    a = accuracy_stats(acc_df())
    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi("Total scans", s["total"], ""), unsafe_allow_html=True)
    c2.markdown(kpi("Helmet / No-helmet", f"{s['h']} / {s['nh']}", "total detections", "#22c55e"), unsafe_allow_html=True)
    c3.markdown(kpi("Labelled accuracy", pct(a["acc"]) if a else "—", "from the Accuracy page", "#f59e0b"),
                unsafe_allow_html=True)
    if df.empty:
        st.info("Nothing to report yet. Run a detection first.")
        return
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download CSV", df.to_csv(index=False).encode(), "detection_history.csv", "text/csv")
    d2.download_button("⬇️ Download HTML report", build_report_html(df, s, a).encode("utf-8"),
                       "smart_helmet_report.html", "text/html")
    st.caption("Tip: open the HTML report in your browser and choose Print → Save as PDF.")
    st.dataframe(df.tail(50).iloc[::-1])


def page_about():
    page_header("ℹ️", "About", "How the system works and what it can't do.")
    st.markdown("""
**Workflow:** webcam/upload → OpenCV frames → (optional low-light enhancement) → trained YOLO helmet detector → Python decision (with temporal smoothing for video/webcam) → virtual ignition display.

**Decision rule:** any no-helmet detection = OFF; helmet detection with no no-helmet detection = ON; no reliable detection = OFF.

**Advanced options:** confidence and IoU thresholds, image size, test-time augmentation, CLAHE low-light enhancement, temporal smoothing, colour-coded boxes, live accuracy testing.

**Tech stack:** Python · YOLO (Ultralytics) · OpenCV · Streamlit · streamlit-webrtc.

**Limitations:** This demo does not prove that a helmet is worn or fastened, and does not associate each helmet with a particular rider. A production system needs rider/helmet association, pillion handling, extensive testing and certified hardware. Never connect this simulation to a real motorcycle.
""")


# ──────────────────────────────────────────────────────────────
# 7. MAIN: sidebar navigation + page routing
# ──────────────────────────────────────────────────────────────
PAGES = {
    "📊 Dashboard": page_dashboard,
    "🖼️ Image Detection": page_image,
    "🎬 Video Detection": page_video,
    "📹 Live Webcam": page_webcam,
    "🔑 Ignition Status": page_ignition,
    "🎯 Accuracy & Model": page_accuracy,
    "🕘 History": page_history,
    "📄 Reports": page_reports,
    "ℹ️ About": page_about,
}

init_state()

with st.sidebar:
    st.markdown('<div class="brand">🏍️ Smart Helmet</div>'
                '<div class="brand-sub">Detection &amp; Ignition Safety Simulation</div>', unsafe_allow_html=True)
    nav = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
    st.markdown("---")
    with st.expander("⚙️ Detection settings"):
        st.slider("Confidence threshold", 0.05, 0.95, DEFAULT_CFG["conf"], 0.05, key="conf",
                  help="Higher = fewer false alarms but may miss riders. Lower = catches more but more mistakes.")
        st.slider("Overlap filter (IoU)", 0.10, 0.90, DEFAULT_CFG["iou"], 0.05, key="iou",
                  help="Removes duplicate boxes on the same object. Lower = stricter.")
        st.selectbox("Inference image size", [320, 416, 512, 640, 800], index=3, key="imgsz",
                     help="Bigger is usually more accurate but slower.")
        st.toggle("Test-time augmentation (slower, often more accurate)", value=False, key="augment")
        st.toggle("Low-light enhancement (CLAHE)", value=False, key="enhance")
        st.slider("Temporal smoothing (frames)", 1, 15, DEFAULT_CFG["smooth_n"], key="smooth_n",
                  help="Video and webcam only. The ignition decision uses a vote over the last N frames so it does not flicker. 1 = off.")
    with st.expander("🎨 Appearance"):
        st.selectbox("Accent theme", list(THEMES.keys()), key="theme")
        st.selectbox("Box style", ["Custom colours", "YOLO default"], key="style")
    with st.expander("🔊 Sound alerts", expanded=True):
        st.toggle("Buzzer when no helmet / voice when helmet", value=True, key="sound")
        t1, t2 = st.columns(2)
        if t1.button("Test buzzer"):
            play_alert("buzzer")
        if t2.button("Test voice"):
            play_alert("speech")
    if MODEL_FILE.exists():
        st.markdown('<span class="badge" style="background:#0f3d2a;color:#4ade80">● Model loaded: best.pt</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge" style="background:#4a1620;color:#f87171">● models/best.pt missing</span>',
                    unsafe_allow_html=True)
    st.caption("Academic simulation only. Not connected to a real bike.")

_a, _b = THEMES[st.session_state.theme]
st.markdown(CSS.replace("__A__", _a).replace("__B__", _b), unsafe_allow_html=True)

PAGES[nav]()
