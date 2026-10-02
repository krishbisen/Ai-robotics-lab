import cv2

from ai_robotics.vision.object_detector import ObjectDetector


detector = ObjectDetector()


image = cv2.imread(
    "projects/object_detection/test.jpg"
)

if image is None:
    raise FileNotFoundError(
        "Could not find test.jpg"
    )


# Run YOLO only once
results = detector.detect(image)


# Extract detection information
detections = detector.get_detections_from_results(
    results
)


for detection in detections:

    print(
        f"Object: {detection['class_name']} | "
        f"Confidence: {detection['confidence']:.2f} | "
        f"Box: {detection['bbox']}"
    )


# Draw detections
annotated_image = results[0].plot()


cv2.imshow(
    "YOLO Object Detection",
    annotated_image
)

cv2.waitKey(0)
cv2.destroyAllWindows()