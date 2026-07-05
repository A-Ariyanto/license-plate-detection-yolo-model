# Author: Abdullah Ariyanto
# This is a script to train a YOLO26 model for license plate recognition using a dataset from Roboflow.

# Before running this script, ensure you have the required packages installed:
# pip3 install ultralytics
# pip3 install roboflow
# pip3 install python-dotenv

import os
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO

load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE = "roboflow-universe-projects"
PROJECT = "license-plate-recognition-rxg4e"
VERSION = 11
EPOCHS = 50
PATIENCE = 10  # Stop early if validation mAP hasn't improved for this many epochs
IMAGE_SIZE = 640
DATA_YAML_PATH = "License-Plate-Recognition-11/data.yaml"

def download_dataset():
    if os.path.exists("License-Plate-Recognition-11"):
        print("Dataset already exists. Skipping download.")
        return "License-Plate-Recognition-11"

    if not ROBOFLOW_API_KEY:
        raise RuntimeError("ROBOFLOW_API_KEY is not set. Copy .env.example to .env and fill in your key.")

    print("Downloading dataset from Roboflow...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION)
    # YOLO26 uses the same label format as YOLOv11, so this download format still works
    dataset = version.download("yolov11")
    return dataset.location  # Path to dataset directory

def train_model():
    print("Training YOLO26 model...")
    # Start from the pretrained nano weights so the model converges with few epochs
    model = YOLO("yolo26n.pt")
    results = model.train(data=DATA_YAML_PATH, epochs=EPOCHS, imgsz=IMAGE_SIZE, patience=PATIENCE)

    print("Evaluating model...")
    results = model.val()

    return results

if __name__ == "__main__":
    dataset_path = download_dataset()
    train_model()
