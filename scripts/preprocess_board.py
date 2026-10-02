import cv2 as cv


IMAGE_PATH = (
    "calibration_images/"
    "archive/data/imgs/leftcamera/"
    "Im_L_1.png"
)


def preprocess(image):

    gray = cv.cvtColor(
        image,
        cv.COLOR_BGR2GRAY
    )

    # Improve local contrast
    clahe = cv.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # Reduce small noise
    blurred = cv.GaussianBlur(
        enhanced,
        (5, 5),
        0
    )

    # Separate black and white regions
    binary = cv.adaptiveThreshold(
        blurred,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        11,
        2
    )

    return gray, enhanced, binary


def main():

    image = cv.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(
            "Image could not be loaded."
        )

    gray, enhanced, binary = preprocess(image)

    cv.imshow("Original", image)
    cv.imshow("Grayscale", gray)
    cv.imshow("Enhanced", enhanced)
    cv.imshow("Binary", binary)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()