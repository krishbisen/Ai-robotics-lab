import numpy as np

from ai_robotics.vision.preprocessing import preprocess


def test_resize():
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    processed = preprocess(image, size=(224, 224))

    assert processed.shape == (224, 224)


def test_grayscale():
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    processed = preprocess(image)

    assert len(processed.shape) == 2


def test_normalization():
    image = np.full((100, 200, 3), 255, dtype=np.uint8)

    processed = preprocess(image)

    assert processed.min() >= 0
    assert processed.max() <= 1