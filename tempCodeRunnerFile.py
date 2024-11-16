from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
from keras.models import model_from_json
import mediapipe as mp

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
json_file = open("model/model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model/model.h5")

# Actions and other variables
actions = ["A", "B", "C", "..."]  # Replace with your action labels
sequence = []
sentence = []
accuracy = []
threshold = 0.8

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Webcam feed
cap = cv2.VideoCapture(0)

def mediapipe_detection(image, hands):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0] * 63)
    return keypoints

def generate_frames():
    global sequence, sentence, accuracy

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        crop_frame = frame[40:400, 0:300]
        frame = cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)
        image, results = mediapipe_detection(crop_frame, hands)
        
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-42:]

        if len(sequence) == 42:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            if res[np.argmax(res)] > threshold:
                action = actions[np.argmax(res)]
                if not sentence or sentence[-1] != action:
                    sentence.append(action)
                    accuracy.append(f"{res[np.argmax(res)] * 100:.2f}%")
                sentence = sentence[-1:]
                accuracy = accuracy[-1:]

        cv2.rectangle(frame, (0, 0), (300, 40), (245, 117, 16), -1)
        cv2.putText(frame, f"Output: {''.join(sentence)} {''.join(accuracy)}", (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
