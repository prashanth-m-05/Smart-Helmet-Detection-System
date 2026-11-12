# Smart Helmet Detection — Laptop-only Simulation

## Features
- Image upload detection
- Uploaded video analysis and annotated MP4 download
- Live webcam detection using WebRTC
- Virtual bike ignition ON/OFF
- CSV detection history

## Requirements
Windows 10/11, Python 3.10 or 3.11 recommended, 16 GB RAM recommended, webcam optional. GPU is not required.

## Windows setup (VS Code terminal)
```powershell
python --version
python -m venv venv
venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python check_gpu.py
```
If PowerShell blocks activation, use Command Prompt:
```cmd
venv\\Scripts\\activate.bat
```

## Add the model
Follow `MODEL_SETUP.md`. The app requires `models/best.pt` from a compatible trained helmet model.

## Run
```bash
streamlit run app.py
```
Open the local address printed in the terminal, normally http://localhost:8501.

## Demo
1. Image tab: upload a rider image.
2. Uploaded video tab: upload a short MP4 and click Process uploaded video.
3. Live webcam tab: click START and allow camera permission.
4. History tab: view/download CSV records.

## Troubleshooting
- `best.pt` missing: follow MODEL_SETUP.md.
- Webcam not opening: allow camera access in browser and Windows settings; close apps using the camera.
- WebRTC blocked: try Chrome/Edge on localhost; use uploaded video as fallback.
- Video won't open: convert to MP4/H.264.
- Slow CPU inference: reduce `imgsz=640` to `imgsz=416` in app.py.
- Package conflicts: create a fresh venv and install only requirements.txt.

## Safety
This is a simulated academic prototype. It does not control a real motorcycle and must never be connected to a real ignition system.
