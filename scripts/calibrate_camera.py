import cv2 as cv
import glob
import numpy as np

from ai_robotics.vision.camera_calibration import calibrate_camera

from pathlib import Path

image_folder = Path("calibration_images/archive/data/imgs/leftcamera")

paths = list(image_folder.glob("*.png"))

print("Images found:", len(paths))

images = []

for path in paths:
    img = cv.imread(path)

    if img is not None:
        images.append(img)


print("Images found:", len(images))


# Calibrate camera
mtx, dist, rvecs, tvecs = calibrate_camera(
    images,
    chessboard_size=(7, 6),
    square_size=25
)


# Print results
print("\nCamera Matrix:")
print(mtx)

print("\nDistortion:")
print(dist)


# Save calibration data
np.savez(
    "camera_calibration.npz",
    camera_matrix=mtx,
    distortion=dist
)


# Test undistortion
img = images[0]

undistorted = cv.undistort(
    img,
    mtx,
    dist,
    None,
    mtx
)

cv.imshow("Original", img)
cv.imshow("Undistorted", undistorted)

cv.waitKey(0)
cv.destroyAllWindows()