import cv2 as cv
from matplotlib.pyplot import gray
import numpy as np


def calibrate_camera(images, chessboard_size=(7, 6), square_size=25):

    objp = np.zeros(
        (chessboard_size[1] * chessboard_size[0], 3),
        np.float32
    )

    objp[:, :2] = np.mgrid[
        0:chessboard_size[0],
        0:chessboard_size[1]
    ].T.reshape(-1, 2)

    objp[:, :2] *= square_size

    objpoints = []
    imgpoints = []

    criteria = (
        cv.TERM_CRITERIA_EPS +
        cv.TERM_CRITERIA_MAX_ITER,
        30,
        0.001
    )

    image_size = None

    for img in images:

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        image_size = gray.shape[::-1]

        ret, corners = cv.findChessboardCornersSB(
            gray, chessboard_size, None
            )

        if ret:

            objpoints.append(objp)
            imgpoints.append(corners)

            print("Chessboard detected")

        else:
            print("Chessboard NOT detected")

    if len(objpoints) == 0:
        raise ValueError(
            "No chessboard corners were detected."
        )

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None
    )


    total_error = 0

    for i in range(len(objpoints)):
        projected_points, _ = cv.projectPoints(
            objpoints[i],
            rvecs[i],
            tvecs[i],
            mtx,
            dist
        )

        error = cv.norm(
            imgpoints[i],
            projected_points,
            cv.NORM_L2
        ) / len(projected_points)

        total_error += error

    mean_error = total_error / len(objpoints)

    print("Mean Reprojection Error:", mean_error)

    return mtx, dist, rvecs, tvecs