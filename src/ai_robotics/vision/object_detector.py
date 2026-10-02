from ultralytics import YOLO


class ObjectDetector:

    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)

    def detect(self, image):
        """
        Run YOLO object detection on an image.
        """
        return self.model(image)

    def get_detections_from_results(self, results):
        """
        Convert YOLO results into simple Python dictionaries.
        """

        detections = []

        for result in results:

            for box in result.boxes:

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                confidence = float(box.conf[0])

                class_id = int(box.cls[0])

                class_name = self.model.names[class_id]

                # Calculate center of bounding box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": (
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2),
                        ),
                        "center": (cx, cy),
                    }
                )

        return detections