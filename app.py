import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model, Model
from collections import deque
import tempfile
import os
import urllib.request

# Auto download face_landmarker.task
FACE_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
FACE_MODEL_PATH = "face_landmarker.task"

if not os.path.exists(FACE_MODEL_PATH):
    with st.spinner("Downloading face landmark model... (30MB, one time only)"):
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
    st.success("Face model downloaded!")

st.set_page_config(
    page_title="DrowsyGuard",
    page_icon="",
    layout="centered"
)

st.title("DrowsyGuard")
st.subheader("Real-Time Driver Drowsiness Detection")
st.markdown("Built using **CNN-LSTM** | CNN Accuracy: **97.94%** | LSTM Accuracy: **99.29%**")
st.divider()

IMG_SIZE = 24
SEQ_LEN  = 10

@st.cache_resource
def load_models():
    cnn  = load_model("models/cnn_eye_model.h5")
    lstm = load_model("models/cnn_lstm_model.h5")
    dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype="float32")
    cnn.predict(dummy, verbose=0)
    extractor = Model(inputs=cnn.inputs, outputs=cnn.layers[-3].output)
    return extractor, lstm

@st.cache_resource
def load_face_detector():
    base_opts = mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
    opts = vision.FaceLandmarkerOptions(
        base_options=base_opts,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return vision.FaceLandmarker.create_from_options(opts)

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def get_eye_roi(frame, landmarks, indices, h, w):
    points   = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x1 = max(min(x_coords) - 10, 0)
    x2 = min(max(x_coords) + 10, w)
    y1 = max(min(y_coords) - 10, 0)
    y2 = min(max(y_coords) + 10, h)
    return frame[y1:y2, x1:x2]

def preprocess_eye(roi):
    if roi is None or roi.size == 0:
        return None
    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    norm    = resized.astype("float32") / 255.0
    return norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)

# Load models
with st.spinner("Loading models..."):
    feature_extractor, lstm_model = load_models()
    face_landmarker = load_face_detector()
st.success("Models loaded successfully!")

# Upload video
st.markdown("### Upload a video to test")
uploaded = st.file_uploader(
    "Upload a driving video (.mp4 / .avi / .mov)",
    type=["mp4", "avi", "mov"]
)

if uploaded:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded.read())
    tfile.flush()

    cap   = cv2.VideoCapture(tfile.name)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 20

    st.markdown("### Detection Results")
    col1, col2, col3 = st.columns(3)
    alert_count  = col1.empty()
    drowsy_count = col2.empty()
    fps_display  = col3.empty()

    frame_placeholder = st.empty()
    progress = st.progress(0)

    feature_buffer = deque(maxlen=SEQ_LEN)
    frame_idx = 0
    alert_c   = 0
    drowsy_c  = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection = face_landmarker.detect(mp_img)

        label = "No Face Detected"
        color = (255, 165, 0)

        if detection.face_landmarks:
            landmarks = detection.face_landmarks[0]
            l_roi = get_eye_roi(frame, landmarks, LEFT_EYE,  h, w)
            r_roi = get_eye_roi(frame, landmarks, RIGHT_EYE, h, w)
            l_inp = preprocess_eye(l_roi)
            r_inp = preprocess_eye(r_roi)

            if l_inp is not None and r_inp is not None:
                l_feat = feature_extractor.predict(l_inp, verbose=0)[0]
                r_feat = feature_extractor.predict(r_inp, verbose=0)[0]
                feature_buffer.append((l_feat + r_feat) / 2.0)

                if len(feature_buffer) == SEQ_LEN:
                    seq  = np.array(list(feature_buffer)).reshape(1, SEQ_LEN, 128)
                    prob = lstm_model.predict(seq, verbose=0)[0][0]

                    if prob < 0.5:
                        label = "DROWSY! ({:.0f}%)".format((1 - prob) * 100)
                        color = (0, 0, 255)
                        drowsy_c += 1
                    else:
                        label = "ALERT ({:.0f}%)".format(prob * 100)
                        color = (0, 200, 0)
                        alert_c += 1
                else:
                    label = "Collecting {}/{}".format(
                        len(feature_buffer), SEQ_LEN)
                    color = (255, 200, 0)

        cv2.rectangle(frame, (0, 0), (w, 55), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(
            frame_rgb, channels="RGB", use_container_width=True)

        progress.progress(min(frame_idx / max(total, 1), 1.0))
        alert_count.metric("Alert Frames",  alert_c)
        drowsy_count.metric("Drowsy Frames", drowsy_c)
        fps_display.metric("FPS", f"{fps:.0f}")

        frame_idx += 1

    cap.release()
    os.unlink(tfile.name)

    st.success("Analysis complete!")
    total_det = alert_c + drowsy_c
    if total_det > 0:
        drowsy_pct = (drowsy_c / total_det) * 100
        if drowsy_pct > 30:
            st.error(
                f"WARNING: Driver was drowsy {drowsy_pct:.1f}% of the time!")
        else:
            st.success(
                f"Driver was mostly alert. Drowsy: {drowsy_pct:.1f}%")

st.divider()
st.markdown("""
**Project Details**
- CNN Eye Classifier Accuracy: 97.94%
- CNN-LSTM Drowsiness Accuracy: 99.29%
- Built by: Santosh Paswan | GEC Vaishali
- Internship: BIT Sindri, Dhanbad
""")