import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import glob
import os
import pandas as pd
import math as mt

model_path = "model_creator/pose_landmarker_full.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

# Placeholder for collected data
attrs_to_get = ["x", "y", "z", "visibility", "presence"]
no_kp_files = []
fail_to_load_files = []

# Specify the directory path containing images
pose_dirs = [p for p in glob.glob("selected_poses/*") if os.path.isdir(p)]
pose_names = [os.path.basename(d) for d in pose_dirs]

df_all = pd.DataFrame({})

for pose in pose_names:
    data = []
    directory_path = f'selected_poses/{pose}'

    # Use glob to get all the image file paths
    image_files = glob.glob(f'{directory_path}/*.jpg')

    # Load the input image from an image file.
    # mp_image = mp.Image.create_from_file('selected_poses/akarna/Screenshot 2024-02-06 at 12.50.57.png')

    with PoseLandmarker.create_from_options(options) as landmarker:
        for image_file in image_files:
            # detect key points. object is pose_landmarker_result.pose_landmarks
            try:
                mp_image = mp.Image.create_from_file(image_file)
                pose_landmarker_result = landmarker.detect(mp_image)
                if len(pose_landmarker_result.pose_landmarks) > 0:
                # extract info out of landmarks object
                    row = []
                    for lmk in attrs_to_get:
                        for x in range(len(pose_landmarker_result.pose_landmarks[0])):
                            row.append(getattr(pose_landmarker_result.pose_landmarks[0][x], lmk))
                    data.append(row)
                else:
                    no_kp_files.append(image_file)
            except:
                fail_to_load_files.append(image_file)
                pass

    columns = []
    for x in range(33):
        for attr in attrs_to_get:
            col = f"kp{x}_{attr}"
            columns.append(col)

    df_pose = pd.DataFrame(data, columns=columns)
    df_pose["pose"] = pose

    if len(df_all) < 1:
        df_all = df_pose.copy()
    else:
        df_all = pd.concat([df_all, df_pose], axis=0)


# User-applied sigmoid on visibility and presence
def sigmoid(x):
  return 1 / (1 + mt.exp(-x))

word1 = 'visibility'
word2 = 'presence'
pattern = f'{word1}|{word2}'
columns_to_modify = df_all.columns[df_all.columns.str.contains(pattern)]
df_all[columns_to_modify] = df_all[columns_to_modify].applymap(sigmoid)

# Creating the csv for the DL Training
df_all.to_csv("pose_landmark_data_10.csv", index=False)


# The path to the directory containing the folders
directory_path = 'selected_poses/'

# List all items in the directory and filter for directories only
folders = [item for item in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, item))]

# Initialize a dictionary to hold the folder names and their item counts
folder_item_counts = {}

# Iterate over each folder and count its contents
for folder in folders:
    folder_path = os.path.join(directory_path, folder)
    item_count = len(os.listdir(folder_path))
    folder_item_counts[folder] = item_count

# Print the count of items in each folder
for folder, count in folder_item_counts.items():
    print(f"{folder}: {count} items")
