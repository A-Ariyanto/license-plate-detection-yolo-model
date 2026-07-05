import os
from ultralytics import YOLO


def test_model(model):
    print("Testing YOLO26 model on a demo")
    
    results = model("image1.jpg", save=True)
    
    print("Results saved")
    
    return results
    
    
if __name__ == "__main__":
    model = YOLO("runs/detect/train/weights/best.pt")
    
    result = test_model(model)
    
    print("Testing completed. Check the output in the 'runs/detect/predict' directory.")