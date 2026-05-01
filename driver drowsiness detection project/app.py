import os
from flask import Flask, render_template, Response, request, jsonify
from camera import VideoCamera
from drowsiness_detector import DrowsinessDetector
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to hold camera and detector state
camera = None
detector = DrowsinessDetector()
current_ear = 0.0
current_perclos = 0.0
current_status = "Normal"

def init_camera(source=0):
    global camera, detector
    if camera:
        camera.release()
    camera = VideoCamera(source)
    detector.reset_stats()

# Initialize with webcam by default
init_camera(0)

def generate_frames():
    global camera, detector, current_ear, current_perclos, current_status
    while True:
        if camera is None:
            break
            
        success, frame = camera.read()
        if not success or frame is None:
            # Avoid CPU busy waiting when video ends or frame is not ready
            import time
            time.sleep(0.1)
            continue
            
        # Process the frame for drowsiness
        processed_frame, ear, perclos, status = detector.process_frame(frame)
        
        # Update globals for the metrics endpoint
        current_ear = ear
        current_perclos = perclos
        current_status = status

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
        # Limit the frame rate to ~30 FPS to prevent overwhelming and crashing the browser
        import time
        time.sleep(0.033)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        init_camera(filepath)
        return jsonify({"message": "File uploaded and processing started"}), 200

@app.route('/set_mode', methods=['POST'])
def set_mode():
    data = request.json
    mode = data.get('mode', 'webcam')
    if mode == 'webcam':
        init_camera(0)
    return jsonify({"message": f"Mode set to {mode}"}), 200

@app.route('/metrics')
def get_metrics():
    global current_ear, current_perclos, current_status
    return jsonify({
        "ear": current_ear,
        "perclos": current_perclos,
        "status": current_status
    })

if __name__ == '__main__':
    app.run(debug=True, threaded=True, use_reloader=False)
