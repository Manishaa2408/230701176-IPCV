import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import winsound
import csv
import os
from datetime import datetime

class DrowsinessDetector:
    def __init__(self):
        self.EAR_THRESHOLD = 0.25
        self.CONSEC_FRAMES = 20
        self.PERCLOS_THRESHOLD = 40

        self.counter = 0
        self.closed_frames = 0
        self.total_frames = 0
        self.drowsy = False

        self.detector = dlib.get_frontal_face_detector()
        # Ensure the model is available
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        self.LEFT_EYE = list(range(36, 42))
        self.RIGHT_EYE = list(range(42, 48))
        
        self.log_file = "detection_logs.csv"
        self._init_log()

    def _init_log(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "EAR", "PERCLOS", "Alert_Status"])

    def log_data(self, ear, perclos, status):
        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"{ear:.2f}", f"{perclos:.2f}", status])

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def process_frame(self, frame):
        self.total_frames += 1
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.array(gray, dtype=np.uint8)

        faces = self.detector(gray)
        
        ear = 0.0
        perclos = 0.0 if self.total_frames == 0 else (self.closed_frames / self.total_frames) * 100
        status = "Normal"

        for face in faces:
            landmarks = self.predictor(gray, face)
            coords = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

            # Draw face box
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw landmarks
            for (x, y) in coords:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            leftEye = coords[self.LEFT_EYE]
            rightEye = coords[self.RIGHT_EYE]

            cv2.polylines(frame, [leftEye], True, (255, 0, 0), 1)
            cv2.polylines(frame, [rightEye], True, (255, 0, 0), 1)

            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            if ear < self.EAR_THRESHOLD:
                self.counter += 1
                self.closed_frames += 1
                self.drowsy = True
            else:
                self.counter = 0
                self.drowsy = False

            perclos = 0.0 if self.total_frames == 0 else (self.closed_frames / self.total_frames) * 100

            if self.counter >= self.CONSEC_FRAMES or perclos > self.PERCLOS_THRESHOLD:
                status = "DROWSY"
                cv2.putText(frame, "DROWSINESS ALERT!", (120, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                # Play sound async
                winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)

            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"PERCLOS: {perclos:.2f}%", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"Status: {status}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if status == "DROWSY" else (0, 255, 0), 2)

        # Log once per frame if a face is detected
        if len(faces) > 0:
            self.log_data(ear, perclos, status)

        return frame, ear, perclos, status

    def reset_stats(self):
        self.counter = 0
        self.closed_frames = 0
        self.total_frames = 0
        self.drowsy = False
