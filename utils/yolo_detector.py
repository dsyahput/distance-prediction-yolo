import cv2
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path, confidence=0.7):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect_objects(self, frame):
        results = self.model(frame, verbose=False, conf=self.confidence)
        return results

    def annotate_frame(self, frame, results):
        return results[0].plot()
