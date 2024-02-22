import pandas as pd
import numpy as np

# constants
VERTICES = {
    'left_shoulder': ['kp13_x', 'kp13_y', 'kp11_x', 'kp11_y', 'kp23_x', 'kp23_y',],
    'right_shoulder': ['kp14_x', 'kp14_y', 'kp12_x', 'kp12_y', 'kp24_x', 'kp24_y',],
    'left_elbow': ['kp11_x', 'kp11_y', 'kp13_x', 'kp13_y', 'kp15_x', 'kp15_y',],
    'right_elbow': ['kp12_x', 'kp12_y', 'kp14_x', 'kp14_y', 'kp16_x', 'kp16_y',],
    'left_wrist': '',
    'right_wrist': '',
    'left_pinky': '',
    'right_pinky': '',
    'left_index': '',
    'left_thumb': '',
    'right_thumb': '',
    'left_hip': ['kp11_x', 'kp11_y', 'kp23_x', 'kp23_y', 'kp25_x', 'kp25_y',],
    'right_hip': ['kp12_x', 'kp12_y', 'kp24_x', 'kp24_y', 'kp26_x', 'kp26_y',],
    'left_knee': ['kp23_x', 'kp23_y', 'kp25_x', 'kp25_y', 'kp27_x', 'kp27_y',],
    'right_knee': ['kp24_x', 'kp24_y', 'kp26_x', 'kp26_y', 'kp28_x', 'kp28_y',],
    'left_ankle': '',
    'right_ankle': '',
    'left_heel': '',
    'right_heel': '',
    'left_foot_index': '',
    'right_foot_index': '',
}
ANGLES_TO_CHECK = {
    'downdog': ['right_shoulder', 'left_shoulder', 'left_elbow', 'right_elbow', 'left_hip', 'right_hip', 'left_knee', 'right_knee'],
    'tree': ['right_shoulder', 'left_shoulder', 'left_elbow', 'right_elbow', 'left_hip', 'right_hip', 'left_knee', 'right_knee'],
    'boat': ['right_shoulder', 'left_shoulder', 'left_elbow', 'right_elbow', 'left_hip', 'right_hip', 'left_knee', 'right_knee'],
    'camel': ['right_shoulder', 'left_shoulder', 'left_elbow', 'right_elbow', 'left_hip', 'right_hip', 'left_knee', 'right_knee'],
    'akarna': ['right_shoulder', 'left_shoulder', 'left_elbow', 'right_elbow', 'left_hip', 'right_hip', 'left_knee', 'right_knee'],
    'warrior': ['right_shoulder', 'left_shoulder', 'left_elbow', 'right_elbow', 'left_hip', 'right_hip', 'left_knee', 'right_knee'],
    'heron': ['right_shoulder', 'left_shoulder', 'left_elbow', 'right_elbow', 'left_hip', 'right_hip', 'left_knee', 'right_knee'],
    'halfmoon': ['right_shoulder', 'left_shoulder', 'left_elbow', 'right_elbow', 'left_hip', 'right_hip', 'left_knee', 'right_knee'],
    'plow': ['right_shoulder', 'left_shoulder', 'left_elbow', 'right_elbow', 'left_hip', 'right_hip', 'left_knee', 'right_knee'],
    'goddess': ['right_shoulder', 'left_shoulder', 'left_elbow', 'right_elbow', 'left_hip', 'right_hip', 'left_knee', 'right_knee'],
    'dance': ['right_shoulder', 'left_shoulder', 'left_elbow', 'right_elbow', 'left_hip', 'right_hip', 'left_knee', 'right_knee'],
    'plank': ['right_shoulder', 'left_shoulder', 'left_elbow', 'right_elbow', 'left_hip', 'right_hip', 'left_knee', 'right_knee'],
    'revolved_triangle': ['right_shoulder', 'left_shoulder', 'left_elbow', 'right_elbow', 'left_hip', 'right_hip', 'left_knee', 'right_knee'],
    'cobra': ['right_shoulder', 'left_shoulder', 'left_elbow', 'right_elbow', 'left_hip', 'right_hip', 'left_knee', 'right_knee']
}

# helper method to get angle out of points a, b, and c which are np.arrays
def get_one_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(cosine_angle))

# main function to retreive all relevant angles for a pose
def get_all_angles(landmarks: list, pose: str) -> list:
    angles = []
