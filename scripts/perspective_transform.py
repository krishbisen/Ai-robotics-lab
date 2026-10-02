import cv2 as cv
import numpy as np
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

PATTERN_SIZE = (7, 6)

INPUT_IMAGE = Path(
    "calibration_images/archive/data/imgs/leftcamera/Im_L_1.png"
)

OUTPUT_IMAGE = Path(
    "perspective_output.png"
)


# ============================================================
# FIND CHESSBOARD INTERNAL CORNERS
# ============================================================

def detect_chessboard(image):
    """
    Detect the internal chessboard corners.

    PATTERN_SIZE = (7, 6)
    means:
        7 corners horizontally
        6 corners vertically
    """

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    found, corners = cv.findChessboardCornersSB(
        gray,
        PATTERN_SIZE,
        flags=(
            cv.CALIB_CB_EXHAUSTIVE |
            cv.CALIB_CB_ACCURACY
        )
    )

    if not found:
        raise RuntimeError(
            "Chessboard corners could not be detected."
        )

    return corners


# ============================================================
# ARRANGE CORNERS INTO A 2D GRID
# ============================================================

def arrange_corners(corners):
    """
    Arrange the 42 detected chessboard corners
    into 6 rows × 7 columns using their image
    coordinates.
    """

    rows = PATTERN_SIZE[1]
    cols = PATTERN_SIZE[0]

    # Convert to (42, 2)
    points = corners.reshape(-1, 2)

    # ------------------------------------------------
    # Step 1: Sort all points according to Y
    # ------------------------------------------------

    points = points[
        np.argsort(points[:, 1])
    ]

    # ------------------------------------------------
    # Step 2: Split into 6 rows
    # ------------------------------------------------

    grid = points.reshape(
        rows,
        cols,
        2
    )

    # ------------------------------------------------
    # Step 3: Sort every row according to X
    # ------------------------------------------------

    for r in range(rows):

        grid[r] = grid[r][
            np.argsort(grid[r, :, 0])
        ]

    return grid.astype(np.float32)

    score_1 = grid_score(grid_1)
    score_2 = grid_score(grid_2)

    if score_1 <= score_2:
        grid = grid_1
    else:
        grid = grid_2

    return grid


# ============================================================
# FIND OUTER CORNERS
# ============================================================

def find_outer_corners(corners):
    """
    Estimate the four outer corners of the chessboard
    from the detected internal corners.
    """

    rows, cols = corners.shape[:2]

    # ------------------------------------------------
    # Create ideal chessboard coordinates
    # for the internal corners
    # ------------------------------------------------

    object_points = []

    for r in range(rows):

        for c in range(cols):

            object_points.append(
                [c, r]
            )

    object_points = np.float32(
        object_points
    )

    image_points = corners.reshape(
        -1,
        2
    ).astype(np.float32)

    # ------------------------------------------------
    # Calculate homography
    # ------------------------------------------------

    H, _ = cv.findHomography(
        object_points,
        image_points
    )

    if H is None:
        raise RuntimeError(
            "Could not calculate homography."
        )

    # ------------------------------------------------
    # Outer board coordinates
    # ------------------------------------------------

    outer_points = np.float32([
        [-1, -1],          # Top-left
        [cols, -1],        # Top-right
        [cols, rows],      # Bottom-right
        [-1, rows]         # Bottom-left
    ]).reshape(
        -1,
        1,
        2
    )

    # ------------------------------------------------
    # Convert ideal coordinates to image coordinates
    # ------------------------------------------------

    outer_corners = cv.perspectiveTransform(
        outer_points,
        H
    )

    return outer_corners.reshape(
        4,
        2
    )


# ============================================================
# PERSPECTIVE TRANSFORMATION
# ============================================================

def perspective_transform(
    image,
    corners,
    output_size=(800, 600)
):
    """
    Convert the chessboard into a bird's-eye view.
    """

    width, height = output_size

    destination_points = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ])

    matrix = cv.getPerspectiveTransform(
        corners.astype(np.float32),
        destination_points
    )

    warped = cv.warpPerspective(
        image,
        matrix,
        (width, height)
    )

    return warped


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = cv.imread(
        str(INPUT_IMAGE)
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not load image:\n{INPUT_IMAGE}"
        )

    print(
        f"Image loaded: {image.shape}"
    )

    # --------------------------------------------------------
    # Detect internal corners
    # --------------------------------------------------------

    corners = detect_chessboard(
        image
    )

    print(
        f"Detected {len(corners)} internal corners."
    )

    # --------------------------------------------------------
    # Arrange corners
    # --------------------------------------------------------

    grid = arrange_corners(
        corners
    )

    print(
        "\nCorner grid:"
    )

    print(
        "Shape:",
        grid.shape
    )

    print(
        "Top-left:",
        grid[0, 0]
    )

    print(
        "Top-right:",
        grid[0, -1]
    )

    print(
        "Bottom-left:",
        grid[-1, 0]
    )

    print(
        "Bottom-right:",
        grid[-1, -1]
    )

    # --------------------------------------------------------
    # Find outer corners
    # --------------------------------------------------------

    outer_corners = find_outer_corners(
        grid
    )

    print(
        "\nEstimated OUTER corners:"
    )

    print(
        "Top-left:",
        outer_corners[0]
    )

    print(
        "Top-right:",
        outer_corners[1]
    )

    print(
        "Bottom-right:",
        outer_corners[2]
    )

    print(
        "Bottom-left:",
        outer_corners[3]
    )

    # --------------------------------------------------------
    # Draw detected outer corners
    # --------------------------------------------------------

    debug_image = image.copy()

    for point in outer_corners:

        x, y = point.astype(int)

        cv.circle(
            debug_image,
            (x, y),
            8,
            (0, 0, 255),
            -1
        )

    cv.imwrite(
        "detected_outer_corners.png",
        debug_image
    )

    # --------------------------------------------------------
    # Perspective transform
    # --------------------------------------------------------

    warped = perspective_transform(
        image,
        outer_corners,
        output_size=(800, 600)
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    cv.imwrite(
        str(OUTPUT_IMAGE),
        warped
    )

    print(
        f"\nPerspective image saved to:"
        f"\n{OUTPUT_IMAGE}"
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    cv.imshow(
        "Original",
        image
    )

    cv.imshow(
        "Detected Outer Corners",
        debug_image
    )

    cv.imshow(
        "Perspective Transform",
        warped
    )

    cv.waitKey(0)
    cv.destroyAllWindows()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()