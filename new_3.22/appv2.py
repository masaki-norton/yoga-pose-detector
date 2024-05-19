import streamlit as st
import cv2
import numpy as np
import joblib

# Initialize the webcam
cap = cv2.VideoCapture(0)  # '0' is typically the default value for the webcam

st.title("Webcam Live Feed")
frame_window = st.image([])

kmeans = joblib.load("pose_estimator_kmeans.pkl")

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture video. Please check your webcam settings.")
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame)

# Release the video capture object
cap.release()
