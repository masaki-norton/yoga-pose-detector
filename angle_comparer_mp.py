import pandas as pd
import numpy as np
from best_poses import *

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
SAMPLE_BEST_DOWNDOG = [
    [0.6849415 , 0.4385808 , 0.47379467],
    [0.6847934 , 0.4198386 , 0.475131  ],
    [0.68384135, 0.42276525, 0.62527484],
    [0.65603083, 0.40431318, 0.57396674],
    [0.6483914 , 0.40910396, 0.45509106],
    [0.61763614, 0.436769  , 0.67206794],
    [0.60486054, 0.43210745, 0.6128188 ],
    [0.71399164, 0.36302376, 0.6027485 ],
    [0.67569697, 0.35926113, 0.49042264],
    [0.7741585 , 0.28499505, 0.6505081 ],
    [0.72596705, 0.2903945 , 0.29765126],
    [0.42484543, 0.6065925 , 0.855795  ],
    [0.42545247, 0.6065874 , 0.82244235],
    [0.5925844 , 0.683898  , 0.78402096],
    [0.591881  , 0.68931067, 0.640119  ],
    [0.7363759 , 0.7676759 , 0.7926514 ],
    [0.71737164, 0.75531054, 0.57859325],
]


# helper method to get angle out of points a, b, and c which are np.arrays
def get_one_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(cosine_angle))

# main function to retreive all relevant angles for a pose
def get_all_angles(landmarks, pose: str) -> list:
    angles = []
    to_check = ANGLES_TO_CHECK[pose] #names of key points, ex: right_shoulder
    for vertex in to_check:
        temp = landmarks[VERTICES[vertex]] #this is the 3 (x,y) of relevant kp, per vertex
        a = np.array(temp[0:2])
        b = np.array(temp[2:4])
        c = np.array(temp[4:])
        angles.append(get_one_angle(a,b,c))
    return angles
