import cv2

from ai_robotics.vision.object_detector import ObjectDetector
from ai_robotics.vision.multi_tracker import MultiObjectTracker


detector = ObjectDetector()
tracker = MultiObjectTracker(max_distance=50)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam")


while True:

    ret, frame = camera.read()

    if not ret:
        break

    # YOLO detection
    results = detector.detect(frame)

    # Convert YOLO results to our dictionaries
    detections = detector.get_detections_from_results(results)

    # Extract centers
    centers = [
        detection["center"]
        for detection in detections
    ]

    # Track multiple objects
    tracked_objects = tracker.update(centers)

    # Draw detections
    for detection in detections:

        x1, y1, x2, y2 = detection["bbox"]
        cx, cy = detection["center"]

        # Find the ID belonging to this center
        object_id = None

        for track_id, center in tracked_objects.items():
            if center == (cx, cy):
                object_id = track_id
                break

        # Bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Center
        cv2.circle(
            frame,
            (cx, cy),
            5,
            (0, 0, 255),
            -1
        )

        # Label
        label = (
            f"ID: {object_id} | "
            f"{detection['class_name']} "
            f"{detection['confidence']:.2f}"
        )

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "YOLO Multi-Object Tracking",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()