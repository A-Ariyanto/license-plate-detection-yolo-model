import os
import cv2
import easyocr
from ultralytics import YOLO

TESTFILES_DIR = "testfiles"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def detect_and_read(model, reader, image_path):
    print(f"\n--- Testing {image_path} ---")

    results = model(image_path, save=True)

    image = cv2.imread(image_path)
    annotated = image.copy()

    plates = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = image[y1:y2, x1:x2]

            # Plates are alphanumeric only, so restrict the character set
            texts = reader.readtext(crop, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            plate_text = "".join(text for _, text, _ in texts)
            for _, text, confidence in texts:
                plates.append(text)
                print(f"Detected plate: {text} (confidence: {confidence:.2f})")

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if plate_text:
                cv2.putText(annotated, plate_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if not plates:
        print("No readable plate text found")

    stem = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(str(results[0].save_dir), f"{stem}_ocr.jpg")
    cv2.imwrite(output_path, annotated)
    print(f"Annotated image with plate text saved to {output_path}")

    return plates


if __name__ == "__main__":
    model = YOLO("runs/detect/train/weights/best.pt")
    reader = easyocr.Reader(["en"])

    for filename in sorted(os.listdir(TESTFILES_DIR)):
        if filename.lower().endswith(IMAGE_EXTENSIONS):
            detect_and_read(model, reader, os.path.join(TESTFILES_DIR, filename))

    print("\nTesting completed. Check the output in the 'runs/detect/predict' directory.")
