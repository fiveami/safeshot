# SafeShot

SafeShot is a PyQt desktop snipping tool that automatically blurs detected faces before saving or copying screenshots.

## Features

- Click-and-drag screen snipping overlay
- Automatic face redaction using multiple detectors:
  - OpenCV DNN (SSD face detector)
  - Optional MTCNN
  - Haar cascade fallback
- Adjustable redaction aggressiveness (`Low`, `Medium`, `High`)
- Optional diagnostics mode for detector debugging
- Save-to-file and copy-to-clipboard flow

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies.
3. Run the app.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Runtime-Only Dependencies

`requirements.txt` currently contains extra packages that are not required to run the app UI.
For day-to-day usage, you can install a smaller set:

```bash
pip install -r requirements-runtime.txt
```

## Project Layout

- `main.py` - app UI, snipping workflow, and image redaction pipeline
- `models/` - OpenCV face detector model files
- `Demo/` - before/after sample images

## Notes

- On first run, if model files are missing, SafeShot can download required OpenCV model assets.
- Face detection is heuristic-based and can miss edge cases (e.g., tiny faces, occlusions, unusual angles).

## Suggested Next Improvements

- Split `main.py` into modules (`ui`, `detectors`, `processing`, `ipc`)
- Add unit tests for bounding-box merge and mask generation behavior
- Move from `print` calls to structured logging
