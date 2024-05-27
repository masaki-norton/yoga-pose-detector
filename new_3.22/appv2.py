import streamlit as st
import cv2
import numpy as np
import joblib
from get_landmarks import get_landmarks_simple, get_landmarks_from_pose
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings

tf.get_logger().setLevel("ERROR")
warnings.filterwarnings('ignore', message="X does not have valid feature names")

# Initialize capture
cap = cv2.VideoCapture(0)

# Load models
model = load_model("nn_v1.h5")
pipeline = joblib.load("pipeline.pkl")

# Initialize frames
st.title("Yoga Pose Estimation")
frame_window = st.image([])
output_box = st.empty()

poses = ['downdog', 'goddess', 'plank', 'tree_chest', 'tree_up', 'warrior2_left', 'warrior2_right']

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture video. Please check your webcam settings.")
        break
    results = get_landmarks_simple(frame, pose)
    if results.pose_landmarks:
        # Drawing on image
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Inference
        lmks = get_landmarks_from_pose(results)
        lmks_transformed = pipeline.transform(np.array([lmks]))
        pred = model.predict(lmks_transformed, verbose=0)

        # print output
        output_box.markdown(
            f"<div style='text-align: center; font-size: 24px;'>"
            f"Predicted Pose: {poses[np.argmax(pred)]}"
            f"</div>",
            unsafe_allow_html=True
        )
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame)

cap.release()
pose.close()
