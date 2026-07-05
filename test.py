import os
import cv2
import easyocr
from ultralytics import YOLO

IMAGE_PATH = "testfiles/image1.jpg"


def test_model(model):
    print("Testing YOLO26 model on a demo")

    results = model(IMAGE_PATH, save=True)

    print("Results saved")

    return results


def read_plates(results):
    print("Reading plate text with EasyOCR")

    reader = easyocr.Reader(["en"])
    image = cv2.imread(IMAGE_PATH)

    plates = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = image[y1:y2, x1:x2]

            # Plates are alphanumeric only, so restrict the character set
            texts = reader.readtext(crop, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            for _, text, confidence in texts:
                plates.append(text)
                print(f"Detected plate: {text} (confidence: {confidence:.2f})")

    if not plates:
        print("No readable plate text found")

    return plates


if __name__ == "__main__":
    model = YOLO("runs/detect/train/weights/best.pt")

    result = test_model(model)

    read_plates(result)

    print("Testing completed. Check the output in the 'runs/detect/predict' directory.")
