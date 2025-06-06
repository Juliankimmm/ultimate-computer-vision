import cv2
import torch
import numpy as np
import mediapipe as mp
from deepface import DeepFace
from collections import deque
import tkinter as tk
from tkinter import ttk
import threading
from PIL import Image, ImageTk

class CVApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Computer Vision Mode Selector")
        self.root.geometry("800x600")
        
        # Current mode
        self.current_mode = "none"
        self.cap = None
        self.running = False
        
        # Initialize models (lazy loading)
        self.yolo_model = None
        self.pose = None
        self.face_cascade = None
        self.emotion_history = deque(maxlen=5)
        
        # Exercise counters
        self.pushup_counter = RepCounter(down_threshold=90, up_threshold=160)
        self.pullup_counter = RepCounter(down_threshold=90, up_threshold=160)
        self.squat_counter = RepCounter(down_threshold=90, up_threshold=160)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title = ttk.Label(main_frame, text="Computer Vision Demo", 
                         font=("Arial", 16, "bold"))
        title.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Mode selection buttons
        ttk.Label(main_frame, text="Select Mode:", font=("Arial", 12)).grid(row=1, column=0, sticky=tk.W, pady=5)
        
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Mode buttons
        modes = [
            ("Object Detection", "object"),
            ("Emotion Recognition", "emotion"), 
            ("Push-up Counter", "pushup"),
            ("Pull-up Counter", "pullup"),
            ("Squat Counter", "squat"),
            ("Stop Camera", "stop")
        ]
        
        for i, (text, mode) in enumerate(modes):
            btn = ttk.Button(button_frame, text=text, 
                           command=lambda m=mode: self.set_mode(m))
            btn.grid(row=i//2, column=i%2, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Status
        self.status_var = tk.StringVar(value="Ready - Select a mode to start")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.grid(row=3, column=0, columnspan=2, pady=20)
        
        # Exercise counters display
        self.counter_frame = ttk.LabelFrame(main_frame, text="Exercise Counters", padding="10")
        self.counter_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        self.pushup_var = tk.StringVar(value="Push-ups: 0")
        self.pullup_var = tk.StringVar(value="Pull-ups: 0") 
        self.squat_var = tk.StringVar(value="Squats: 0")
        
        ttk.Label(self.counter_frame, textvariable=self.pushup_var).grid(row=0, column=0, padx=10)
        ttk.Label(self.counter_frame, textvariable=self.pullup_var).grid(row=0, column=1, padx=10)
        ttk.Label(self.counter_frame, textvariable=self.squat_var).grid(row=0, column=2, padx=10)
        
        # Reset counters button
        ttk.Button(self.counter_frame, text="Reset Counters", 
                  command=self.reset_counters).grid(row=1, column=0, columnspan=3, pady=10)
        
        # Instructions
        instructions = ttk.LabelFrame(main_frame, text="Instructions", padding="10")
        instructions.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        inst_text = """• Select a mode above to start the camera with that specific function
• Only one mode runs at a time for optimal performance  
• Press 'q' in the camera window to return to mode selection
• Exercise counters persist until manually reset"""
        
        ttk.Label(instructions, text=inst_text, justify=tk.LEFT).grid(row=0, column=0)
    
    def set_mode(self, mode):
        if self.running:
            self.stop_camera()
        
        if mode == "stop":
            self.current_mode = "none"
            self.status_var.set("Camera stopped")
            return
            
        self.current_mode = mode
        self.status_var.set(f"Loading {mode} mode...")
        self.root.update()
        
        # Start camera in separate thread
        threading.Thread(target=self.start_camera, daemon=True).start()
    
    def load_models_for_mode(self, mode):
        """Lazy load only the models needed for current mode"""
        if mode == "object" and self.yolo_model is None:
            self.status_var.set("Loading YOLO model...")
            self.root.update()
            self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.yolo_model.conf = 0.4
            self.yolo_model.iou = 0.45
            
        elif mode == "emotion" and self.face_cascade is None:
            self.status_var.set("Loading face detection...")
            self.root.update()
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
        elif mode in ["pushup", "pullup", "squat"] and self.pose is None:
            self.status_var.set("Loading pose estimation...")
            self.root.update()
            mp_pose = mp.solutions.pose
            self.pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
    
    def start_camera(self):
        self.load_models_for_mode(self.current_mode)
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_var.set("❌ Cannot open camera")
            return
        
        self.running = True
        self.status_var.set(f"Running {self.current_mode} mode - Press 'q' to stop")
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            
            # Process frame based on current mode
            if self.current_mode == "object":
                frame = self.process_object_detection(frame)
            elif self.current_mode == "emotion":
                frame = self.process_emotion_recognition(frame)
            elif self.current_mode in ["pushup", "pullup", "squat"]:
                frame = self.process_exercise_counter(frame)
            
            cv2.imshow(f"CV Demo - {self.current_mode.title()} Mode", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        self.stop_camera()
    
    def process_object_detection(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.yolo_model(img_rgb, size=640)
        detections = results.xyxy[0]
        
        for *box, conf, cls in detections.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)
            label = self.yolo_model.names[class_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            caption = f"{label} {conf:.2f}"
            cv2.putText(frame, caption, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def process_emotion_recognition(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            x2, y2 = x + w, y + h
            face_img = frame[y:y2, x:x2]
            
            try:
                analysis = DeepFace.analyze(face_img, actions=["emotion"], enforce_detection=False)
                dominant_emotion = analysis["dominant_emotion"]
            except:
                dominant_emotion = "N/A"
                
            self.emotion_history.append(dominant_emotion)
            
            cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)
            
            if len(self.emotion_history) > 0:
                common_emotion = max(set(self.emotion_history), key=self.emotion_history.count)
            else:
                common_emotion = dominant_emotion
                
            cv2.putText(frame, common_emotion, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return frame
    
    def process_exercise_counter(self, frame):
        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = self.pose.process(img_rgb)
        
        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark
            
            def landmark_to_point(idx):
                lm = landmarks[idx]
                return np.array([int(lm.x * w), int(lm.y * h)])
            
            # Draw skeleton
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results_pose.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            
            # Calculate angles and update counters
            if self.current_mode in ["pushup", "pullup"]:
                shoulder = landmark_to_point(12)
                elbow = landmark_to_point(14) 
                wrist = landmark_to_point(16)
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                
                if self.current_mode == "pushup":
                    count = self.pushup_counter.update(elbow_angle)
                    self.pushup_var.set(f"Push-ups: {count}")
                    cv2.putText(frame, f"Push-ups: {count} | Angle: {elbow_angle:.1f}°", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:  # pullup
                    count = self.pullup_counter.update(elbow_angle)
                    self.pullup_var.set(f"Pull-ups: {count}")
                    cv2.putText(frame, f"Pull-ups: {count} | Angle: {elbow_angle:.1f}°",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    
            elif self.current_mode == "squat":
                hip = landmark_to_point(24)
                knee = landmark_to_point(26)
                ankle = landmark_to_point(28) 
                knee_angle = calculate_angle(hip, knee, ankle)
                count = self.squat_counter.update(knee_angle)
                self.squat_var.set(f"Squats: {count}")
                cv2.putText(frame, f"Squats: {count} | Angle: {knee_angle:.1f}°",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        return frame
    
    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.status_var.set("Camera stopped - Select a mode to restart")
    
    def reset_counters(self):
        self.pushup_counter = RepCounter(down_threshold=90, up_threshold=160)
        self.pullup_counter = RepCounter(down_threshold=90, up_threshold=160) 
        self.squat_counter = RepCounter(down_threshold=90, up_threshold=160)
        self.pushup_var.set("Push-ups: 0")
        self.pullup_var.set("Pull-ups: 0")
        self.squat_var.set("Squats: 0")
    
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        self.stop_camera()
        self.root.destroy()


class RepCounter:
    def __init__(self, down_threshold, up_threshold):
        self.down_thresh = down_threshold
        self.up_thresh = up_threshold
        self.state = "UP"
        self.count = 0

    def update(self, angle):
        if self.state == "UP" and angle < self.down_thresh:
            self.state = "DOWN"
        elif self.state == "DOWN" and angle > self.up_thresh:
            self.state = "UP"
            self.count += 1
        return self.count


def calculate_angle(a, b, c):
    """Computes the angle at point b formed by the points a‐b‐c (in degrees)."""
    a = np.array(a)
    b = np.array(b) 
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


if __name__ == "__main__":
    app = CVApp()
    app.run()