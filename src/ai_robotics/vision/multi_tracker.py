import math


class MultiObjectTracker:

    def __init__(self, max_distance=50):
        self.objects = {}
        self.next_id = 1
        self.max_distance = max_distance

    def calculate_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2

        return math.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )

    def update(self, centers):

        updated_objects = {}

        for center in centers:

            best_id = None
            best_distance = self.max_distance

            for object_id, previous_center in self.objects.items():

                distance = self.calculate_distance(
                    center,
                    previous_center
                )

                if distance < best_distance:
                    best_distance = distance
                    best_id = object_id

            if best_id is None:

                best_id = self.next_id
                self.next_id += 1

            updated_objects[best_id] = center

        self.objects = updated_objects

        return self.objects