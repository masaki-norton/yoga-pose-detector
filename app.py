import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
import av
import cv2
from angle_comparer import angle_comparer
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import joblib
from best_poses import *
import time
from collections import deque
from PIL import Image
import queue


# ======================== Setup and Model Loading =========================

# Make page wide (remove default wasted whitespace)
#Remove the menu button and Streamlit icon on the footer
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
print('hiding default menu')
st.markdown(hide_default_format, unsafe_allow_html=True)

#Change font to Catamaran type
streamlit_style = """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Catamaran:wght@100;200;300;400;500;600;700;800;900&display=swap');

            html, body, [class*="css"]  {
            font-family: 'Catamaran', sans-serif;
            }
            </style>
            """
st.markdown(streamlit_style, unsafe_allow_html=True)


# Centralize the title 'Hatha Project'
st.markdown("<h3 style='text-align: center; color: black;'>🧘‍♀️ Practice Session 🧘‍♀️</h1>", unsafe_allow_html=True)


# Load Model and Scaler
interpreter = tf.lite.Interpreter(model_path="model_creator/3.tflite")
interpreter.allocate_tensors()
model = tf.keras.models.load_model('model_creator/24112023_sub_model.h5')
scaler = joblib.load('model_creator/scaler.pkl')

# Define necessary dictionaries
label_mapping = {
    0: 'Downdog',
    1: 'Goddess',
    2: 'Plank',
    3: 'Plank',
    4: 'Tree',
    5: 'Tree',
    6: 'Warrior',
    7: 'Warrior'}
best_pose_map = {
    0: best_downdog,
    1: best_goddess,
    2: best_plank_elbow,
    3: best_plank_straight,
    4: best_tree_chest,
    5: best_tree_up,
    6: best_warrior,
    7: best_warrior}
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}
landmark_dict = {
    'landmarks_left_elbow': (9, 7, 5),
    'landmarks_right_elbow': (10, 8, 6),
    'landmarks_left_shoulder': (11, 5, 7),
    'landmarks_right_shoulder': (12, 6, 8),
    'landmarks_hip_left': (13, 11, 5),
    'landmarks_hip_right': (14, 12, 6),
    'landmarks_left_knee': (15, 13, 11),
    'landmarks_right_knee': (16, 14, 12)}
lm_list = list(landmark_dict.keys())
lm_points = list(landmark_dict.values())
joint_dict = {'landmarks_left_elbow': 'left elbow',
              'landmarks_right_elbow': 'left elbow',
              'landmarks_left_shoulder': 'left shoulder',
              'landmarks_right_shoulder': 'right shoulder',
              'landmarks_hip_left': 'left hip',
              'landmarks_hip_right': 'left hip',
              'landmarks_left_knee': 'left knee',
              'landmarks_right_knee': 'right knee'
              }


# The result_queue variable will store the value for each iteration of the callback. These are the
# values that will be later printed below the livestream. str and dict are put as hints/expectations of the values.
result_queue: queue.Queue[str | dict] = queue.Queue()

# ==================== Functions definition and Variables =====================

# Defining functions for the score evaluation for the live pose.
def get_score_eval(score: float):
    if score <= 0.6:
        return "bad"
    elif score <= 0.85:
        return "good"
    else:
        return "perfect!"

# This function draws the key points on key bodyparts. These are then used to
# calculate body angles with the function below. In the live video, only the
# bodyparts that construct the worst angle are shown.
def draw_key_points(frame, keypoints, conf_threshold):
    max_dim = max(frame.shape)
    shaped = np.squeeze(np.multiply(keypoints, [max_dim,max_dim,1]))
    print(int(shaped[0][0]))
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > conf_threshold:
            cv2.circle(frame,(int(kx), int(ky)-80), 1, (90,195,217), 15)
    return frame


# This function draws lines between the key bodyparts. In the live video, only the
# bodyparts that constructi the worst angle are shown.
def draw_connections(frame, keypoints, edges, confidence_threshold):
    max_dim = max(frame.shape)
    shaped = np.squeeze(np.multiply(keypoints, [max_dim,max_dim,1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)-80), (int(x2), int(y2)-80), (88, 235, 52), 3)



# This function takes a (3,17) landmarks array based on the body position on
# livestream and returns the softmax output from the multiclass classification pose NN.
def get_pose(landmarks: list):

    # Preparation of input before feeding the model.
    lms_51 = np.array(landmarks).reshape(51).tolist()
    landmarks_array = np.array(lms_51).reshape(1, -1)
    landmarks_array = np.delete(landmarks_array, np.arange(2, landmarks_array.size, 3))
    landmarks_array = landmarks_array[np.newaxis, :]
    scaled_landmarks = scaler.transform(landmarks_array)

    # Feed landmarks_array to model to get softmax output.
    prediction = model.predict(scaled_landmarks)
    return prediction

# These are the variables that contain a list of values that are used to
# calculate the average for each shown value. Each list is as long as the
# window size

window_size = 15  # Number of frames to average over
score_angles_history = deque(maxlen=window_size)
angle_diff_history = deque(maxlen=window_size)
avg_percentage_diff_history = deque(maxlen=window_size)
worst_name_history = deque(maxlen=window_size)
average_score_history = deque(maxlen=window_size)
pose_history = deque(maxlen=window_size)

# Defining the angle_comparer to create video and overlay
def callback(frame):

    # the global command takes the created variables from outside the function
    # and makes them valid on the inside. All changes to them inside this
    # function will affect the variable outside the function as well.
    global worst_name_history, angle_diff_history, avg_percentage_diff_history, score_angles_history, average_score_history

    s_time = time.time()
    """ ======== 1. Movenet to get Landmarks ======== """

    # the variable image is an array depiction of the livestream feed frame in
    # colour arrangement bgr24.
    image = frame.to_ndarray(format="bgr24")

    # The image gets resized to for interpreter and the array type changed to
    # float.
    img = tf.image.resize_with_pad(np.expand_dims(image, axis=0), 192, 192)
    input_image = tf.cast(img, dtype=tf.float32)

    We need to load
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]["index"], input_image.numpy())

    # Run inference
    interpreter.invoke()

    # Get the output details and retrieve the keypoints with scores
    output_details = interpreter.get_output_details()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]["index"])
    # print(keypoints_with_scores[0][0][:, :2])
    # print(type(keypoints_with_scores))

    """ ======== 2. Pose Prediction ======== """
    pose_output = get_pose(keypoints_with_scores[0][0])
    target_pose = label_mapping[np.argmax(pose_output)]
    if (keypoints_with_scores[0][0][:, 2]).min() < 0.1 or np.max(pose_output) < 0.90:
        target_pose = "...still thinking..."

    result_queue.put(target_pose)
    pose_history.append(target_pose)
    # print(pose_history)


    """ ======== 3. Scoring of Pose ====+===="""

    best = np.array(best_pose_map[np.argmax(pose_output)])
    test_angle_percentage_diff, average_percentage_diff, score_angles, score_angles_unscaled, average_score = angle_comparer(keypoints_with_scores[0][0][:, :2], best)

    best = np.array(best_pose_map[np.argmax(pose_output)])
    test_angle_percentage_diff, average_percentage_diff, score_angles, score_angles_unscaled, average_score = angle_comparer(keypoints_with_scores[0][0][:, :2], best)
    average_score = 1-sum(test_angle_percentage_diff)/len(test_angle_percentage_diff)
    index_of_worst = test_angle_percentage_diff.index(max(test_angle_percentage_diff))
    worst_points = lm_points[index_of_worst]
    result_queue.put(lm_list[index_of_worst])
    average_score_history.append(average_score)

    worst_kps = []
    for i in lm_points[index_of_worst]:
        worst_kps.append((np.squeeze(keypoints_with_scores)[i]).tolist())

    worst_edges = {
    (worst_points[0], worst_points[1]): None,
    (worst_points[1], worst_points[2]): None,
    }

    result_queue.put(worst_edges)
    # average_score_history.append(1-average_score)
    worst_name_history.append(lm_list[index_of_worst])
    # print(worst_name_history)
    # sliding_avg_score = np.mean(average_score_history, axis=0)

    if np.max(pose_output) > 0.5 and np.max(pose_output) < 0.95:
        # Draw the landmarks onto the image with threshold
        draw_key_points(image, worst_kps, conf_threshold=0.2)
        draw_connections(image, keypoints_with_scores, worst_edges, 0.2)

    mirrored_image = cv2.flip(image, 1)

    return av.VideoFrame.from_ndarray(mirrored_image, format="bgr24")




# ==================== Actual UI output =====================

@st.cache_data
def load_and_cache_image(image_path):
    return Image.open(image_path)

best_all_poses = load_and_cache_image('mika_poses/all_poses.jpeg')

# Container for Images
images_container = st.container()
with images_container:
    st.image(best_all_poses, use_column_width=True)

video_analysis_container = st.container()
with video_analysis_container:
    # Code for live video feed & pose analysis
    webrtc_streamer(
        key="example",
        video_frame_callback=callback,
        rtc_configuration={  # Add this line
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False}  # Disable audio
    )

labels_placeholder = st.empty()
angle_perc = st.empty()
timecount =  st.empty()

col1, col2, col3 = st.columns(3)
column1 = col1.empty()
column2 = col2.empty()
column3 = col3.empty()

# Add the footer with copyright information
st.markdown("<div style='text-align: center; color: grey;'>Copyright © The Hatha Team 2023</div>", unsafe_allow_html=True)


while True:
    s_time = time.time()
    result = max(result_queue.get())
    result2 = max(result_queue.get())
    result3 = max(result_queue.get())
    total_score = sum(average_score_history)  # Sum up the values in the deque
    average_score = total_score / window_size

    your_pose = max(set(pose_history))
    fix_your = joint_dict[max(set(worst_name_history), key=worst_name_history.count)]
    your_score = get_score_eval(average_score)

    # If statements for the text colours
    if average_score <= 0.6:
        result = "bad"
        result_color = "red"
    elif average_score <= 0.85:
        result = "good"
        result_color = "orange"
    else:
        result = "perfect!"
        result_color = "green"

    column1.markdown(f"<p style='font-size:50px; color:black;'>Pose</p>\n<p style='font-size:35px; font-weight:bold; color:black;'>{your_pose}</p>", unsafe_allow_html=True)

    # Show label and result in black
    column2.markdown(f"<p style='font-size:50px; color:black;'>Score</p>\n<p style='font-size:35px; font-weight:bold; color:{result_color};'>{result}</p>", unsafe_allow_html=True)

    # # Show label and result in black
    column3.markdown(f"<p style='font-size:50px; color:black;'>Fix your</p>\n<p style='font-size:35px; font-weight:bold; color:red;'>{fix_your}</p>", unsafe_allow_html=True)
