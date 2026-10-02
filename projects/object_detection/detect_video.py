import cv2

from ai_robotics.vision.object_detector import ObjectDetector


# Create detector
detector = ObjectDetector()


# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam")


while True:

    # Read one frame
    success, frame = camera.read()

    if not success:
        print("Could not read frame")
        break

    # Run YOLO
    results = detector.detect(frame)

    # Draw detections
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow(
        "YOLO Webcam Detection",
        annotated_frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release resources
camera.release()
cv2.destroyAllWindows()