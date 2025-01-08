from flask import Flask, render_template, jsonify, request, url_for
from flask_socketio import SocketIO
from flask_socketio import emit
import cv2
from ultralytics import YOLO
import threading
import time
import os
import json
from collections import Counter

app = Flask(__name__, static_folder='static', template_folder='static', static_url_path='')
socketio = SocketIO(app)

# Load video and model
video_path = os.path.join('static/processed', 'traffic_video_processed.mp4')
model = YOLO("yolo11n.pt")

latest_detections = []
detection_running = False
lane_occupancy = {f'lane{i}': 'Clear' for i in range(1, 5)}

# Function to run object detection
# def detect_objects():

# Function to initialize processing on Flask start
def initialize_video_processing():
    originals_path = 'static/originals'
    processed_path = 'static/processed'

    # Check if paths exist
    if not os.path.exists(originals_path):
        print(f"Error: Originals path '{originals_path}' does not exist.")
        return
    if not os.path.exists(processed_path):
        print(f"Error: Processed path '{processed_path}' does not exist.")
        os.makedirs(processed_path)

    # Get list of .mp4 files in originals and processed folders
    originals = [f for f in os.listdir(originals_path) if f.endswith('.mp4')]
    processed = [f for f in os.listdir(processed_path) if f.endswith('.mp4') or f.endswith('.json')]

    # Check for missing processed videos
    for original_file in originals:
        base_name = os.path.splitext(original_file)[0]
        processed_video = f"{base_name}_processed.mp4"
        label_json = f"{base_name}_processed.json"

        if processed_video not in processed or label_json not in processed:
            original_path = os.path.join(originals_path, original_file)
            processed_video_path = os.path.join(processed_path, processed_video)
            label_json_path = os.path.join(processed_path, label_json)

            print(f"Processing missing videos for: {original_file}")
            process_and_generate_videos(original_path, processed_video_path, label_json_path)

# Function to process video and generate processed and label files
def process_and_generate_videos(original_path, processed_video_path, label_json_path, target_fps=24):
    if not os.path.exists(original_path):
        socketio.emit('log_message', {'message': f"Error: Video file '{original_path}' does not exist."})
        return

    cap = cv2.VideoCapture(original_path)
    if not cap.isOpened():
        socketio.emit('log_message', {'message': f"Error: Unable to open video file '{original_path}'."})
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, round(original_fps / target_fps))  # Interval to skip frames

    processed_out = cv2.VideoWriter(
        processed_video_path,
        cv2.VideoWriter_fourcc(*'avc1'),
        target_fps,
        (frame_width, frame_height)
    )

    conf_threshold = 0.3
    all_detections = []
    frame_count = 0  # Initialize frame_count to 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = []
        frame_objects = Counter()

        # Process every `frame_interval` frame
        if frame_count % frame_interval == 0:
            detections = []

            # YOLO 모델 설정 변경
            try:
                results = model.predict(
                    source=frame,
                    conf=conf_threshold,
                    iou=0.45,
                    max_det=50
                )
            except Exception as e:
                print(f"Error in YOLO model prediction: {e}")
                break

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    class_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())

                    class_name = model.names[class_id]
                    vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
                    if class_name in vehicle_classes:
                        detections.append({
                            'class': class_name,
                            'confidence': round(conf, 3),
                            'box': [int(x1), int(y1), int(x2), int(y2)]
                        })
                        frame_objects[class_name] += 1

                        # 객체 감지된 영역 표시
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f'{class_name} {conf:.2f}', (int(x1), int(y1)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Append detections to all_detections
            all_detections.append({
                'frame': int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                'detections': detections
            })

            # Write processed frame
            processed_out.write(frame)

        frame_count += 1

    cap.release()
    processed_out.release()

    # Save detections to JSON
    with open(label_json_path, 'w') as json_file:
        json.dump(all_detections, json_file, indent=4)

    print(f"Processed video saved to: {processed_video_path}")
    print(f"Label data saved to: {label_json_path}")

@app.route('/')
def index():
    return render_template('homepage.html')

@app.route('/index')
def render_index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

# 영상 업로드 시 실행되는 함수
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"message": "No video part in the request"}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400
    
    # 파일 이름 설정 및 저장 경로 지정
    filename = file.filename
    original_path = os.path.join('static/originals', filename)
    
    try:
        # 파일을 static/originals 폴더에 저장
        file.save(original_path)
        
        # 분석 및 처리 경로 설정
        base_name = os.path.splitext(filename)[0]
        processed_video_path = os.path.join('static/processed', f"{base_name}_processed.mp4")
        label_json_path = os.path.join('static/processed', f"{base_name}_processed.json")
        
        # 분석 및 처리 함수 호출
        process_and_generate_videos(original_path, processed_video_path, label_json_path)
        
        return jsonify({
            "message": "File uploaded and processing completed successfully",
            "processed_video": f"/static/processed/{base_name}_processed.mp4",
            "label_data": f"/static/processed/{base_name}_processed.json"
        }), 200
    except Exception as e:
        return jsonify({"message": f"Failed to upload and process file: {e}"}), 500

# 영상 목록을 불러오는 함수
@app.route('/get_video_list')
def get_video_list():
    try:
        video_files = os.listdir('static/processed')
        video_files = [f for f in video_files if f.endswith('.mp4')]
        return jsonify({"videos": video_files}), 200
        # URL 생성
        video_urls = [url_for('static', filename=f'processed/{f}') for f in video_files]
        return jsonify({"videos": video_urls}), 200
    except Exception as e:
        return jsonify({"message": f"Failed to get videos: {e}"}), 500

@app.route('/get_label_data/<video_name>')
def get_label_data(video_name):
    label_path = os.path.join(f'static/processed/{video_name}.json')
    print(label_path)
    if not os.path.exists(label_path):
        return jsonify({"message": "Label file not found"}), 404
    
    try:
        with open(label_path, 'r') as f:
            label_data = json.load(f)
        return jsonify(label_data), 200
    except Exception as e:
        return jsonify({"message": f"Error reading label file: {e}"}), 500

if __name__ == '__main__':
    socketio.run(app, debug=True)
    
initialize_video_processing()


