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
2. Fine-tune the pretrained `yolo26n` (nano) model for 3 epochs at 640×640.
3. Run validation and save everything under `runs/detect/train/`.

Epochs, image size, and dataset version are constants at the top of `train.py` — bump `EPOCHS` up for a more serious run once everything works.

## Testing

```bash
python3 test.py
```

Loads the best weights from training (`runs/detect/train/weights/best.pt`) and runs detection on a sample image. Annotated output is saved to `runs/detect/predict/`. Sample images and a test video live in `testfiles/`.

## Project structure

```
├── train.py        # Downloads the dataset and trains the model
├── test.py         # Runs the trained model on sample media
├── testfiles/      # Sample images and video for testing
├── .env.example    # Template for your Roboflow API key
└── .gitignore      # Excludes dataset, runs, weights, and .env
```
