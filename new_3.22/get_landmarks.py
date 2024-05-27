import cv2
import mediapipe as mp
import numpy as np

def get_landmarks(image: np.ndarray) -> list[float]:
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = []
        for lmk in results.pose_landmarks.landmark:
            # landmarks.extend([lmk.x, lmk.y, lmk.z, lmk.visibility])
            landmarks.extend([lmk.x, lmk.y])
        return landmarks
    else:
        return None

def get_landmarks_simple(image: np.ndarray, pose) -> mp.solutions.pose.PoseLandmark:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return pose.process(image_rgb)

def get_landmarks_from_pose(results) -> list[float]:
    landmarks = []
    for lmk in results.pose_landmarks.landmark:
        # landmarks.extend([lmk.x, lmk.y, lmk.z, lmk.visibility])
        landmarks.extend([lmk.x, lmk.y])
    return landmarks


def main() -> None:
    image_path = "clean_data/TEST_TRAIN/downdog/00000128.jpg"
    image = cv2.imread(image_path)
    print(get_landmarks(image))
    return None

if __name__ == "__main__":
    main()
