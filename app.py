from flask import Flask, render_template, Response, stream_with_context, request, jsonify
from flask_cors import CORS
import cv2
import time
from ultralytics import YOLO
import torch

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# setup gpu
# device = '1' if torch.cuda.is_available() else 'cpu'
# if device == '1':
#     torch.cuda.set_device(1)

# print('device:', device)

# load model and class
model = YOLO("models/infusion-drop.pt")
class_name = ["drop"]
confidence = 0.5

video_capture = None
last_drop_time = 0
total_drops = 0
drops_in_one_minute = 0
start_time = time.time()


def detect_drops(frame):
    results = model.predict(frame)[0]
    detections = []
    for result in results.boxes:
        x1, y1, x2, y2 = result.xyxy[0]
        confidence = result.conf[0]
        class_id = result.cls[0]
        detections.append({
            'x1': int(x1),'y1': int(y1),
            'x2': int(x2),'y2': int(y2),
            'confidence': float(confidence),
            'class_id': int(class_id),
            'class_name': model.names[int(class_id)]
        })
    return detections

def count_total_drops(frame):
    detections = detect_drops(frame)
    return len(detections)

def process_frame(frame):
    global total_drops, last_drop_time, drops_in_one_minute, start_time

    drop_count = count_total_drops(frame)
    total_drops += drop_count

    current_time = time.time()
    time_diff = current_time - start_time
    if time_diff >= 60:
        drops_in_one_minute = total_drops - drops_in_one_minute
        start_time = current_time

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global video_capture, total_drops, drops_in_one_minute, start_time
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        return jsonify({"message": "Error: Unable to access the camera."})
    total_drops = 0
    drops_in_one_minute = 0
    start_time = time.time()

    def generate_frames():
        global video_capture
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            detections = detect_drops(frame)
            for detection in detections:
                x1, y1, x2, y2 = detection['x1'], detection['y1'], detection['x2'], detection['y2']
                confidence = detection['confidence']
                class_name = detection['class_name']
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # RGB format
                label = f"{class_name} ({confidence:.2f})"
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (int(x1), int(y1) - label_height - 10), (int(x1) + label_width, int(y1)), (0, 255, 0), -1)
                cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            process_frame(frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n')

    return Response(stream_with_context(generate_frames()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/stop_camera", methods=["POST"])
def stop_detection():
    global video_capture

    if video_capture is not None:
        video_capture.release()
        cv2.destroyAllWindows() 
        video_capture = None

    global total_drops, drops_in_one_minute, start_time
    total_drops = 0
    drops_in_one_minute = 0
    start_time = time.time()

    return jsonify({"message": "Object detection and camera stopped."})

# cv2.destroyAllWindows() 


@app.route("/drop_stats", methods=["GET"])
def get_drop_stats():
    global total_drops, drops_in_one_minute

    return jsonify({"total_drops": total_drops, 
                    "drops_in_one_minute": drops_in_one_minute})

if __name__ == "__main__":
    app.run(port=8901)
