
from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

app = Flask(__name__)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def fingers_up(lm):
    fingers = []
    fingers.append(lm[4].x < lm[3].x)
    for tip in [8, 12, 16, 20]:
        fingers.append(lm[tip].y < lm[tip - 2].y)
    return fingers

def detect_gesture(lm):
    f = fingers_up(lm)
    gestures = {
        (0,0,0,0,0): "Fist ✊",
        (1,1,1,1,1): "Open Palm ✋",
        (1,0,0,0,0): "Thumbs Up 👍",
        (0,1,0,0,0): "One ☝️",
        (0,1,1,0,0): "Two ✌️",
        (0,1,1,1,0): "Three 🤟",
        (0,1,1,1,1): "Four ✋",
        (1,1,0,0,1): "Rock 🤘",
        (1,0,0,0,1): "Call Me 🤙",
        (1,1,0,0,0): "OK 👌",
        (0,1,0,0,1): "Point 👉",
        (0,1,1,0,1): "Peace ✌️",
        (0,0,1,1,1): "Stop ✋",
        (0,0,0,0,1): "Pinky ☝️",
        (1,1,1,0,0): "Gun 👉"
    }
    return gestures.get(tuple(f), "Detecting...")
def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gestures = []

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                g = detect_gesture(hand.landmark)
                gestures.append(g)

        # Display both gestures
        y = 40
        for i, g in enumerate(gestures):
            cv2.putText(frame, f"Hand {i+1}: {g}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            y += 40

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
