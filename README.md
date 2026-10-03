# AI Robotics Lab

A practical learning and experimentation repository focused on **Machine Learning, Computer Vision, and robotics-oriented perception**.

This repository is where I turn previously learned ML/CV concepts into reusable, testable components and small robotics-oriented experiments.

## What I Have Learned and Practiced

### Machine Learning Engineering
- Built a reusable ML training pipeline.
- Used configuration files to keep experiment parameters separate from code.
- Used scikit-learn pipelines for preprocessing and model training.
- Evaluated classification models using Accuracy, Precision, Recall, F1 score, and Confusion Matrix.
- Saved and loaded trained models using Joblib.
- Separated training and inference into reusable scripts.
- Added pytest tests for ML utilities and inference.

### Computer Vision
- Image preprocessing: resizing, grayscale conversion, and normalization.
- Camera calibration using chessboard images.
- Camera matrix and distortion estimation.
- Image undistortion.
- Perspective transformation using manually selected points.
- Object detection with YOLO.
- Webcam and video-based detection.
- Centroid-based object tracking.
- Multi-object tracking using centroid matching.
- ByteTrack-based object tracking.
- Pixel-to-world coordinate mapping using a homography.

## Main Work Completed

### 1. ML Training and Inference
Located in `projects/ml_inference/`.

The project demonstrates:

`Data → Preprocessing → Training → Evaluation → Model Saving → Inference`

The trained Iris classifier uses scikit-learn and is saved with Joblib.

Reusable components:
- `src/ai_robotics/ml/model_loader.py`
- `src/ai_robotics/ml/evaluation.py`

### 2. Image Preprocessing
Located in `src/ai_robotics/vision/preprocessing.py`.

Supports resizing, grayscale conversion, and normalization.

Tests:
`tests/test_preprocessing.py`

### 3. Camera Calibration
Camera calibration was implemented using OpenCV and chessboard images.

The process estimates the camera matrix, lens distortion coefficients, rotation/translation information, and reprojection error.

Calibration parameters:
`camera_calibration.npz`

Implementation:
`src/ai_robotics/vision/camera_calibration.py`

### 4. Perspective Transformation
Perspective transformation was implemented using four manually selected image points.

Implementation:
`scripts/perspective_transform.py`

### 5. YOLO Object Detection
Located in `projects/object_detection/`.

Includes image detection, video detection, webcam detection, and tracking experiments.

Reusable detector:
`src/ai_robotics/vision/object_detector.py`

Current model:
`yolo11n.pt`

### 6. Object Tracking
#### Centroid Tracker
`src/ai_robotics/vision/tracker.py`

Tracks an object's center between frames and calculates its movement.

#### Multi-Object Tracker
`src/ai_robotics/vision/multi_tracker.py`

Maintains multiple object IDs by comparing current object centers with previously detected centers.

### 7. ByteTrack
ByteTrack was integrated through the Ultralytics tracking interface.

Example:
`projects/object_detection/bytetrack_webcam.py`

### 8. Pixel to World Coordinate Mapping
Located in `projects/coordinate_mapping/pixel_to_world.py`.

A homography-based mapping converts image pixel coordinates into coordinates on a known planar workspace.

## Repository Structure

~~~text
Ai-robotics-lab/
├── README.md
├── .gitignore
├── pyproject.toml
├── calibration_images/
│   └── archive/
├── camera_calibration.npz
├── yolo11n.pt
├── configs/
│   └── ml_config.json
├── model/
│   └── iris_model.pkl
├── projects/
│   ├── ml_inference/
│   │   ├── train.py
│   │   └── inference.py
│   ├── object_detection/
│   │   ├── detect_image.py
│   │   ├── detect_video.py
│   │   ├── tracker_webcam.py
│   │   ├── multi_tracker_webcam.py
│   │   └── bytetrack_webcam.py
│   └── coordinate_mapping/
│       └── pixel_to_world.py
├── scripts/
│   ├── calibrate_camera.py
│   ├── perspective_transform.py
│   └── preprocess_board.py
├── src/
│   └── ai_robotics/
│       ├── ml/
│       │   ├── evaluation.py
│       │   └── model_loader.py
│       ├── vision/
│       │   ├── preprocessing.py
│       │   ├── camera_calibration.py
│       │   ├── object_detector.py
│       │   ├── tracker.py
│       │   └── multi_tracker.py
│       └── utils/
└── tests/
    ├── test_model_loader.py
    ├── test_inference.py
    ├── test_preprocessing.py
    ├── test_tracker.py
    └── test_multi_tracker.py
~~~

## Testing

The repository uses **pytest** for testing reusable components.

~~~bash
python -m pytest
~~~

Tests cover ML utilities, inference, image preprocessing, and tracking components.

## Installation

Install the project in editable mode:

~~~bash
python -m pip install -e .
~~~

Core project dependencies currently include Python 3.13+, scikit-learn, and Joblib.

The computer-vision experiments additionally use OpenCV and Ultralytics.

## Tools and Technologies

- Python
- OpenCV
- NumPy
- scikit-learn
- Joblib
- Ultralytics YOLO
- ByteTrack
- pytest
- Git / GitHub

## Learning Approach

Each experiment is used to understand:
1. The problem
2. The underlying concept
3. A practical implementation
4. How to make the implementation reusable
5. How to test important parts
6. How the component can fit into a robotics system

The long-term direction is to connect **AI + Computer Vision + robotics** into practical systems.

## Current Status

Completed practical components include:
- ML training and inference pipeline
- Model saving/loading
- ML evaluation utilities
- Image preprocessing
- Camera calibration
- Perspective transformation
- YOLO object detection
- Centroid tracking
- Multi-object tracking
- ByteTrack integration
- Pixel-to-world coordinate mapping
- Automated tests for core components

The repository will continue to grow as new robotics and perception experiments are learned and implemented.
