import math
from sklearn.metrics.pairwise import euclidean_distances
import cv2

def calculate_angle(a, b):
    delta_x = a[0] - b[0]
    delta_y = a[1] - b[1]
    angle = math.atan2(delta_y, delta_x) * (180 / math.pi)
    return angle

def calculate_slope_m(a, b):
    slope = abs((a[1]-b[1])/(a[0]-b[0]))
    return slope

def calculate_perimeter(points, s, l):
    # Ensure s and l are within the bounds of the points array
    if s < 0 or l >= len(points) or s > l:
        raise ValueError("Invalid indices")

    total = 0
    for x in range(s, l):
        total += euclidean_distances([points[x]], [points[x + 1]])
    total += euclidean_distances([points[s]], [points[l]])
    return total