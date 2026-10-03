import cv2
import numpy as np


image_points = np.float32([
    [100, 100],
    [600, 100],
    [600, 400],
    [100, 400]
])

world_points = np.float32([
    [0, 0],
    [60, 0],
    [60, 40],
    [0, 40]
])


H = cv2.getPerspectiveTransform(
    image_points,
    world_points
)


def pixel_to_world(x, y, H):
    pixel = np.float32([[[x, y]]])

    world = cv2.perspectiveTransform(
        pixel,
        H
    )

    world_x = world[0][0][0]
    world_y = world[0][0][1]

    return world_x, world_y


world_x, world_y = pixel_to_world(350, 250, H)

print("World coordinate:", world_x, world_y)