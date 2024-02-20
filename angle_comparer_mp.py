import pandas as pd
import numpy as np

# helper method to get angle out of points a, b, and c which are np.arrays
def get_one_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(cosine_angle))

# main function to retreive all relevant angles for a pose
def get_all_angles(landmarks: list, pose: str) -> list:
    angles = []
