from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
import cv2
from ultralytics import YOLO
import threading
import time
import os
from collections import Counter
from pymongo import MongoClient

app = Flask(__name__)
socketio = SocketIO(app)

# MongoDB URI 설정(not used)
mongo_uri = os.getenv('MONGO_URI')
client = MongoClient(mongo_uri)
db = client['TrafficAccident']  # 데이터베이스 선택

model = YOLO('yolov8n.pt')

# Load video and model
video_path = os.path.join('static', 'traffic_video.mp4')

# 영상 목록에서 선택한 영상의 경로를 저장하는 함수
def select_video_path(selected_filename):
    global video_path 
    video_path = os.path.join('static', selected_filename)

latest_detections = []
detection_running = False
lane_occupancy = {f'lane{i}': 'Clear' for i in range(1, 5)}

# Function to run object detection
def detect_objects():
    global latest_detections, detection_running, lane_occupancy
    cap = cv2.VideoCapture(video_path)
    
    conf_threshold = 0.3
    
    while detection_running:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        # 데이터 초기화 - 여기로 이동
        detections = []
        frame_objects = Counter()
        
        # YOLO 모델 설정 변경
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            iou=0.45,
            max_det=50
        )
        
        # 감지된 객체 처리 방식 개선
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
        
        # 차선 분석
        frame_width = frame.shape[1]
        lane_width = frame_width // 4
        lane_objects = {f'lane{i}': 0 for i in range(1, 5)}
        
        for detection in detections:
            box = detection['box']
            center_x = (box[0] + box[2]) / 2
            lane_index = int(center_x // lane_width) + 1
            
            if 1 <= lane_index <= 4:
                lane_objects[f'lane{lane_index}'] += 1
        
        # 차선 혼잡도 계산
        for lane, count in lane_objects.items():
            threshold = 3
            lane_occupancy[lane] = 'Congested' if count >= threshold else 'Clear'
        
        try:
            # 데이터 준비 및 전송
            update_data = {
                'detections': detections,
                'object_counts': dict(frame_objects),
                'lane_occupancy': lane_occupancy
            }
            
            # 데이터 로깅
            print("Sending detection update:", update_data)
            
            # Socket.IO 이벤트 발생
            socketio.emit('detection_update', update_data, namespace='/')
            
        except Exception as e:
            print(f"Error sending detection update: {e}")
        
        time.sleep(0.03)
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('start-detection')
def handle_start_detection():
    global detection_running
    if not detection_running:
        detection_running = True
        detection_thread = threading.Thread(target=detect_objects, daemon=True)
        detection_thread.start()

@socketio.on('pause-detection')
def handle_pause_detection():
    global detection_running
    detection_running = False

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
    file_path = os.path.join('static', filename)
    
    try:
        # 파일을 static 폴더에 저장
        file.save(file_path)
        return jsonify({"message": "File uploaded successfully"}), 200
    except Exception as e:
        return jsonify({"message": f"Failed to upload file: {e}"}), 500

# 영상 목록을 불러오는 함수
@app.route('/get_video_list')
def get_videos():
    try:
        video_files = os.listdir('static')
        video_files = [f for f in video_files if f.endswith('.mp4')]
        return jsonify({"videos": video_files}), 200
    except Exception as e:
        return jsonify({"message": f"Failed to get videos: {e}"}), 500

if __name__ == '__main__':
    socketio.run(app, debug=True)

