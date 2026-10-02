import cv2

from ai_robotics.vision.object_detector import ObjectDetector
from ai_robotics.vision.tracker import CentroidTracker


# Create detector and tracker
detector = ObjectDetector()
tracker = CentroidTracker()


# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam")


while True:

    # Read frame
    ret, frame = camera.read()

    if not ret:
        print("Could not read frame")
        break


    # -----------------------------
    # 1. YOLO DETECTION
    # -----------------------------

    results = detector.detect(frame)


    # Convert YOLO results into dictionaries
    detections = detector.get_detections_from_results(results)


    # -----------------------------
    # 2. PROCESS DETECTIONS
    # -----------------------------

    for detection in detections:

        # Get bounding box
        x1, y1, x2, y2 = detection["bbox"]

        # Get object center
        cx, cy = detection["center"]


        # -----------------------------
        # 3. TRACK MOVEMENT
        # -----------------------------

        tracking_result = tracker.update((cx, cy))

        dx, dy = tracking_result["movement"]


        # -----------------------------
        # 4. DRAW BOUNDING BOX
        # -----------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # -----------------------------
        # 5. DRAW CENTER
        # -----------------------------

        cv2.circle(
            frame,
            (cx, cy),
            5,
            (0, 0, 255),
            -1
        )


        # -----------------------------
        # 6. CREATE TEXT
        # -----------------------------

        label = (
            f"{detection['class_name']} "
            f"{detection['confidence']:.2f}"
        )

        movement_text = (
            f"Move: ({dx}, {dy})"
        )


        # -----------------------------
        # 7. DISPLAY OBJECT LABEL
        # -----------------------------

        cv2.putText(
            frame,
            label,
            (x1, y1 - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


        # -----------------------------
        # 8. DISPLAY MOVEMENT
        # -----------------------------

        cv2.putText(
            frame,
            movement_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2
        )


    # -----------------------------
    # 9. SHOW FRAME
    # -----------------------------

    cv2.imshow(
        "YOLO Object Tracking",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# 10. CLEANUP
# -----------------------------

camera.release()
cv2.destroyAllWindows()