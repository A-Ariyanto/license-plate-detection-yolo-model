# License Plate Detection with YOLO26

A learning project for training a YOLO model to detect and identify car license plates — including at hard angles, where many detectors on the market still struggle.

This started out as an old YOLOv11 experiment (hence the repo name) and has been revived and upgraded to **YOLO26**, the latest Ultralytics model. The main goal is to learn the full workflow: pulling a dataset, training a model, and running it on real images and video.

## Dataset

The model is trained on the [License Plate Recognition dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e) from Roboflow Universe (workspace `roboflow-universe-projects`, version 11). The dataset is downloaded automatically the first time you run `train.py`.

## Running on Google Colab (recommended)

This project is meant to be run on Google Colab with a **T4 GPU** — training is far faster there than on a laptop, and the T4 handles the nano model at 640×640 comfortably.

1. In Colab, go to **Runtime → Change runtime type** and select **T4 GPU**.
2. Store your Roboflow API key in Colab's Secrets (the 🔑 icon in the left sidebar) as `ROBOFLOW_API_KEY`, with notebook access enabled. This keeps the key out of the notebook itself.
3. Run the following in notebook cells:

   ```python
   !git clone https://github.com/A-Ariyanto/license-plate-detection-yolo-v11.git
   %cd license-plate-detection-yolo-v11
   !pip install ultralytics roboflow python-dotenv easyocr
   ```

   ```python
   import os
   from google.colab import userdata
   os.environ["ROBOFLOW_API_KEY"] = userdata.get("ROBOFLOW_API_KEY")

   !python train.py
   !python test.py
   ```

   `train.py` reads the key from the environment, so no `.env` file is needed on Colab.

4. Download `runs/detect/train/weights/best.pt` before the runtime disconnects — Colab wipes the filesystem when the session ends.

## Setup (local)

1. Install the dependencies:

   ```bash
   pip3 install ultralytics roboflow python-dotenv easyocr
   ```

2. Create your `.env` file from the template and add your Roboflow API key (find it in your [Roboflow account settings](https://app.roboflow.com/settings/api)):

   ```bash
   cp .env.example .env
   ```

   ```
   ROBOFLOW_API_KEY=your_roboflow_api_key_here
   ```

   The `.env` file is gitignored, so your key stays out of version control.

## Training

```bash
python3 train.py
```

This will:

1. Download the dataset from Roboflow (skipped if `License-Plate-Recognition-11/` already exists).
2. Fine-tune the pretrained `yolo26n` (nano) model for up to 50 epochs at 640×640, with early stopping (`patience=10`) so it quits once validation mAP stops improving.
3. Run validation and save everything under `runs/detect/train/`.

Epochs, patience, image size, and dataset version are constants at the top of `train.py`.

## Testing

```bash
python3 test.py
```

Loads the best weights from training (`runs/detect/train/weights/best.pt`) and runs detection + EasyOCR on every image in `testfiles/`, saving a `<name>_ocr.jpg` with the plate text drawn on it. Annotated output is saved to `runs/detect/predict*/`. Video files are skipped — per-frame processing takes too long for a quick test cycle.

## Test cycle reports

### Test cycle 1 — 2026-07-05

First full run of the pipeline (train → validate → detect → OCR) on Google Colab with a T4 GPU.

**Training** — `yolo26n` fine-tuned at 640×640; stopped at 10 epochs (~25 minutes) as 50 takes too long. Validation on 2,048 images / 2,195 plate instances:

| Metric | Result |
| --- | --- |
| Precision | 0.953 |
| Recall | 0.917 |
| mAP50 | 0.961 |
| mAP50-95 | 0.669 |
| Inference speed | ~3 ms/image (T4) |

The metrics are strong for a nano model: it finds ~92% of plates and 95% of its detections are correct.

**Detection + OCR demo** — `test.py` on `testfiles/image1.jpg` produced **no detections**, so the EasyOCR step had nothing to read. The plate ("XDC 896") is clearly visible in the image, so the likely causes are:

1. The image is very wide (2:1), so YOLO inferred it at 320×640 — shrinking the plate to roughly 60×30 px, near the lower limit of what the model saw in training.
2. It is an Australian-style plate, while the training data is mostly US/international, which may push confidence below the default 0.25 threshold.
3. And the plate is at a hard angle and black plate which may be harder for yolo to detect.

**Next steps for cycle 2:**

- Re-run inference with `imgsz=1280` and `conf=0.1` to confirm the resolution/confidence theory.
- Test on `image2.jpg`, `image3.jpg`, and `videotest.mp4`.
- If low resolution is confirmed as the cause, bake `imgsz=1280` into `test.py`.

### Test cycle 2 — 2026-07-05

Ran the updated `test.py` (now loops over all files in `testfiles/`) against the same cycle-1 weights. Video skipped for now.

| Image | Detection | OCR result | Actual plate | Verdict |
| --- | --- | --- | --- | --- |
| `image1.jpg` | ❌ none | — | XDC 896 | Same failure as cycle 1 |
| `image2.jpg` | ✅ 1 plate | "CHE 88X" (conf 0.83–0.88) | CME 88X | 1 character wrong (M→H) |
| `image3.jpg` | ✅ 1 plate | "HENZUI" (conf 0.02) | AXM 43J | Garbage — crop too small/dark |

**Takeaways:**

- The detector works: 2 of 3 plates found, and both misses/weak results have clear explanations. `image2` and `image3` are portrait shots where the plate stays reasonably large at 640px; `image1` is the 2:1 wide image where the plate shrinks to ~60px (the cycle-1 theory still stands — the `imgsz=1280` experiment hasn't been run yet).
- The remaining errors are mostly **OCR-side, not detection-side**. On `image2` the box was right and EasyOCR was confident but misread one character; on `image3` the box was right but the cropped plate is too small and dark to read. More YOLO training won't fix these.
- Note: the model is **not** trained from scratch — `train.py` starts from pretrained `yolo26n.pt` weights and fine-tunes. Reaching mAP50 0.96 in 10 epochs is only possible because of that head start.

**Next steps for cycle 3:**

- Actually run the `imgsz=1280` / `conf=0.1` experiment for `image1.jpg` (carried over from cycle 2's plan).
- Improve the OCR input: upscale the plate crop 2–4× and boost contrast before passing it to EasyOCR — this should fix both the M→H misread and the unreadable `image3` crop.
- Longer term: train longer (raise `patience`) or move up to `yolo26s` for hard angles.

### Where I left off (paused 2026-07-05)

Current state: training works well (mAP50 0.96), detection works on 2 of 3 test images, OCR reads plates but makes character-level mistakes on small/dark crops. The trained weights (`best.pt`) live only on Colab — re-download or re-train (~25 min on a T4) when resuming.

To continue, pick up the cycle-3 next steps above. The two concrete code changes queued up for `test.py`:

1. Pass `imgsz=1280` (and optionally a lower `conf`) to the model call so wide images like `image1.jpg` keep the plate large enough to detect.
2. Preprocess the plate crop before EasyOCR: upscale 2–4× and boost contrast.

## Project structure

```
├── train.py        # Downloads the dataset and trains the model
├── test.py         # Runs the trained model on sample media
├── testfiles/      # Sample images and video for testing
├── .env.example    # Template for your Roboflow API key
└── .gitignore      # Excludes dataset, runs, weights, and .env
```
