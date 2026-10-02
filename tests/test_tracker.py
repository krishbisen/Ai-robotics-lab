from ai_robotics.vision.tracker import CentroidTracker


def test_first_update():
    tracker = CentroidTracker()

    result = tracker.update((100, 100))

    assert result["center"] == (100, 100)
    assert result["movement"] == (0, 0)


def test_movement():
    tracker = CentroidTracker()

    tracker.update((100, 100))

    result = tracker.update((120, 130))

    assert result["movement"] == (20, 30)