from ultralytics import YOLO


class Detector:
    def __init__(self, model_path="yolov8n.pt"):
        # Load YOLO model
        self.model = YOLO(model_path)

        # COCO class IDs we care about
        self.target_classes = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
            0: "person"
        }

    def detect(self, frame):
        results = self.model(frame, verbose=False)

        detections = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])

                if cls_id in self.target_classes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])

                    detections.append({
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "confidence": conf,
                        "class": self.target_classes[cls_id]
                    })

        return detections