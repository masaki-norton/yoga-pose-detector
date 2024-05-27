import streamlit as st
import cv2
import numpy as np
import joblib
from get_landmarks import get_landmarks_simple
import mediapipe as mp

cap = cv2.VideoCapture(0)

st.title("Yoga Pose Estimation")
frame_window = st.image([])

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
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame)

cap.release()
pose.close()
