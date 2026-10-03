import cv2
import numpy as np
from ultralytics import YOLO

from ai_robotics.vision.object_detector import ObjectDetector
from ai_robotics.vision.object_detector import ObjectDetector


# -----------------------------
# Pixel → World function
# -----------------------------

def pixel_to_world(x, y, H):
    pixel = np.float32([[[x, y]]])

    world = cv2.perspectiveTransform(
        pixel,
        H
    )

    world_x = world[0][0][0]
    world_y = world[0][0][1]

    return world_x, world_y


# -----------------------------
# Calibration points
# -----------------------------

image_points = np.float32([
    [100, 100],
    [600, 100],
    [600, 400],
    [100, 400]
])

world_points = np.float32([
    [0, 0],
    [60, 0],
    [60, 40],
    [0, 40]
])

H = cv2.getPerspectiveTransform(
    image_points,
    world_points
)


# -----------------------------
# YOLO + ByteTrack
# -----------------------------

model = YOLO("yolo11n.pt")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam")


while True:

    ret, frame = camera.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml"
    )

    boxes = results[0].boxes

    if boxes.id is not None:

        for box, track_id, class_id, confidence in zip(
            boxes.xyxy,
            boxes.id,
            boxes.cls,
            boxes.conf
        ):

            x1, y1, x2, y2 = box.tolist()

            # Object center in pixels
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Pixel → World
            world_x, world_y = pixel_to_world(
                cx,
                cy,
                H
            )

            class_name = model.names[int(class_id)]
            track_id = int(track_id)
            confidence = float(confidence)

            print(
                f"ID: {track_id} | "
                f"{class_name} | "
                f"Pixel: ({cx}, {cy}) | "
                f"World: ({world_x:.2f}, {world_y:.2f})"
            )

            # Draw bounding box
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

            # Draw center
            cv2.circle(
                frame,
                (cx, cy),
                5,
                (0, 0, 255),
                -1
            )

            label = (
                f"ID:{track_id} "
                f"{class_name} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            world_text = (
                f"World: ({world_x:.1f}, {world_y:.1f}) cm"
            )

            cv2.putText(
                frame,
                world_text,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                2
            )

    cv2.imshow(
        "YOLO + ByteTrack + World Coordinates",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()