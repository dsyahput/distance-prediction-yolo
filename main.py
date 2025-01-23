import cv2
from utils.yolo_detector import YOLODetector
from utils.polynomial_regression import PolynomialRegression

def main():

    x = [1512, 979, 680, 511, 384, 308, 265, 211, 180, 156, 136, 119, 102, 93, 87, 81, 75] # Area of bounding box
    y = [1.0, 1.25, 1.50, 1.75, 2.0, 2.25, 2.50, 2.75, 3.0, 3.25, 3.50, 3.75, 4.0, 4.25, 4.50, 4.75, 5.0] # Distance in meters

    yolo_detector = YOLODetector('models/best.pt', confidence=0.7)
    poly_regressor = PolynomialRegression(x, y)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        results = yolo_detector.detect_objects(frame)
        annotated_frame = yolo_detector.annotate_frame(frame, results)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = yolo_detector.model.names[class_id]

                if class_name == 'Ball':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1) / 10.0 

                    distance = poly_regressor.predict_distance(area)
                    
                    cv2.putText(annotated_frame, f"Ball Distance: {distance:.2f} m", 
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Webcam Inference', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
