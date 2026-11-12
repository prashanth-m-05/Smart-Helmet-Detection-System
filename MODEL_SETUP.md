# Required helmet model setup

The application code is included, but trained helmet weights are not bundled. The app needs a helmet-specific YOLO model saved as `models/best.pt`. A generic COCO YOLO checkpoint is not a helmet/no-helmet classifier.

## Recommended: train your own model
1. Visit https://universe.roboflow.com/innovatech/motorcycle-helmet-q0wmd
2. Review dataset classes, quality and license/attribution terms.
3. Export/download in YOLOv8 format.
4. Extract the dataset so `dataset/data.yaml` exists. Keep the image and label folders at the paths specified inside data.yaml.
5. Open `train.py` and adjust the DATA path if needed.
6. Run `python train.py`.
7. When training finishes, the script copies weights to `models/best.pt`.
8. Restart the app with `streamlit run app.py`.

CPU training can take a long time. For a first test, use a small dataset and fewer epochs (for example, 10) then increase later.

## Compatible class names
The app recognises names containing `helmet` as helmet, except names containing `no helmet`, `without helmet`, `nohelmet`, or `withouthelmet`, which are treated as no-helmet. If your dataset uses other labels, update `classify_name()` in app.py.

A helmet object detector does not prove that the rider is wearing or fastening the helmet. This is a classroom simulation only.
