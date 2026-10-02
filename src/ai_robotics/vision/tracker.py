class CentroidTracker:

    def __init__(self):
        self.previous_center = None

    def update(self, current_center):
        if self.previous_center is None:
            self.previous_center = current_center

            return {
                "center": current_center,
                "movement": (0, 0)
            }

        previous_x, previous_y = self.previous_center
        current_x, current_y = current_center

        dx = current_x - previous_x
        dy = current_y - previous_y

        self.previous_center = current_center

        return {
            "center": current_center,
            "movement": (dx, dy)
        }