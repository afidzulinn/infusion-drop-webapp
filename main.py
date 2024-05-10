import os
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import cv2
import time
import torch
import uvicorn
from ultralytics import YOLO

## windows
# import pathlib
# from pathlib import Path
# pathlib.PosixPath = pathlib.WindowsPath


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("models/infusion-drop.pt")

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

@app.get("/")
async def check_health():
    return "Healthy"

@app.post("/access_camera")
async def start_detection(background_tasks: BackgroundTasks):
    global video_capture, total_drops, last_drop_time, drops_in_one_minute, start_time

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        return {"message": "Error: Unable to access the camera."}

    total_drops = 0
    drops_in_one_minute = 0
    start_time = time.time()

    def detect_objects():
        global video_capture
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            process_frame(frame)
            time.sleep(0.1)

        cv2.destroyAllWindows()

    background_tasks.add_task(detect_objects)

    return {"message": "Object detection started."}

@app.post("/stop_camera")
async def stop_detection():
    global video_capture

    if video_capture is not None:
        video_capture.release()
        video_capture = None

    global total_drops, drops_in_one_minute, start_time
    total_drops = 0
    drops_in_one_minute = 0
    start_time = time.time()

    return {"message": "Object detection and camera stopped."}

@app.get("/drop_stats")
async def get_drop_stats():
    global total_drops, drops_in_one_minute

    return {"total_drops": total_drops, "drops_in_one_minute": drops_in_one_minute}

if __name__ == "__main__":
    uvicorn.run(host='0.0.0.0', port=8501, reload=True)
