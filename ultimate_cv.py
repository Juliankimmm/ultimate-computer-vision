import cv2
import torch
import numpy as np
import mediapipe as mp
from deepface import DeepFace
from collections import deque
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import threading
from PIL import Image, ImageTk
import json
import os
from datetime import datetime
import pickle

class CVApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Advanced Computer Vision Suite")
        self.root.geometry("900x700")
        
        # Current mode
        self.current_mode = "none"
        self.cap = None
        self.running = False
        
        # Initialize models (lazy loading)
        self.yolo_model = None
        self.pose = None
        self.face_cascade = None
        self.hands = None
        self.emotion_history = deque(maxlen=5)
        
        # User corrections for learning
        self.user_corrections = self.load_corrections()
        self.correction_mode = False
        self.selected_box = None
        self.frame_for_correction = None
        
        # Exercise counters
        self.pushup_counter = RepCounter(down_threshold=90, up_threshold=160)
        self.pullup_counter = RepCounter(down_threshold=90, up_threshold=160)
        self.squat_counter = RepCounter(down_threshold=90, up_threshold=160)
        
        # Face recognition database
        self.face_database = self.load_face_database()
        
        # Color tracking
        self.color_ranges = {
            'red': [(0, 50, 50), (10, 255, 255), (170, 50, 50), (180, 255, 255)],
            'blue': [(100, 50, 50), (130, 255, 255)],
            'green': [(40, 50, 50), (80, 255, 255)],
            'yellow': [(20, 50, 50), (30, 255, 255)],
            'orange': [(5, 150, 150), (20, 255, 255)],
            'purple': [(130, 50, 50), (160, 255, 255)],
            'cyan':   [(80, 100, 100), (100, 255, 255)],
            'magenta': [(140, 50, 50), (170, 255, 255)]
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Main tab
        main_tab = ttk.Frame(notebook)
        notebook.add(main_tab, text="Main Functions")
        self.setup_main_tab(main_tab)
        
        # Advanced tab
        advanced_tab = ttk.Frame(notebook)
        notebook.add(advanced_tab, text="Advanced Features")
        self.setup_advanced_tab(advanced_tab)
        
        # Settings tab
        settings_tab = ttk.Frame(notebook)
        notebook.add(settings_tab, text="Settings & Learning")
        self.setup_settings_tab(settings_tab)
    
    def setup_main_tab(self, parent):
        main_frame = ttk.Frame(parent, padding="10")
        main_frame.pack(expand=True, fill='both')
        
        # Title
        title = ttk.Label(main_frame, text="Computer Vision Demo Suite", 
                         font=("Arial", 16, "bold"))
        title.pack(pady=(0, 20))
        
        # Mode selection buttons
        ttk.Label(main_frame, text="Core Functions:", font=("Arial", 12, "bold")).pack(anchor='w', pady=5)
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        # Core mode buttons
        core_modes = [
            ("Smart Object Detection", "object"),
            ("Emotion Recognition", "emotion"), 
            ("Push-up Counter", "pushup"),
            ("Pull-up Counter", "pullup"),
            ("Squat Counter", "squat")
        ]
        
        for i, (text, mode) in enumerate(core_modes):
            btn = ttk.Button(button_frame, text=text, 
                           command=lambda m=mode: self.set_mode(m))
            btn.grid(row=i//2, column=i%2, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Status
        self.status_var = tk.StringVar(value="Ready - Select a mode to start")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.pack(pady=20)
        
        # Exercise counters display
        counter_frame = ttk.LabelFrame(main_frame, text="Exercise Counters", padding="10")
        counter_frame.pack(fill='x', pady=10)
        
        self.pushup_var = tk.StringVar(value="Push-ups: 0")
        self.pullup_var = tk.StringVar(value="Pull-ups: 0") 
        self.squat_var = tk.StringVar(value="Squats: 0")
        
        counters_row = ttk.Frame(counter_frame)
        counters_row.pack()
        
        ttk.Label(counters_row, textvariable=self.pushup_var).pack(side='left', padx=20)
        ttk.Label(counters_row, textvariable=self.pullup_var).pack(side='left', padx=20)
        ttk.Label(counters_row, textvariable=self.squat_var).pack(side='left', padx=20)
        
        ttk.Button(counter_frame, text="Reset Counters", 
                  command=self.reset_counters).pack(pady=10)
    
    def setup_advanced_tab(self, parent):
        advanced_frame = ttk.Frame(parent, padding="10")
        advanced_frame.pack(expand=True, fill='both')
        
        # Title
        ttk.Label(advanced_frame, text="Advanced CV Features", 
                 font=("Arial", 16, "bold")).pack(pady=(0, 20))
        
        # Advanced mode buttons
        advanced_modes = [
            ("Hand Gesture Detection", "hand_gesture"),
            ("Color Object Tracking", "color_tracking"),
            ("Motion Detection", "motion_detection"),
            ("Distance Measurement", "distance_measure")
        ]
        
        button_frame = ttk.Frame(advanced_frame)
        button_frame.pack(pady=10)
        
        for i, (text, mode) in enumerate(advanced_modes):
            btn = ttk.Button(button_frame, text=text, 
                           command=lambda m=mode: self.set_mode(m))
            btn.grid(row=i//3, column=i%3, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Stop button
        ttk.Button(advanced_frame, text="Stop Camera", 
                  command=lambda: self.set_mode("stop")).pack(pady=20)
        
        # Instructions for advanced features
        instructions = ttk.LabelFrame(advanced_frame, text="Advanced Feature Instructions", padding="10")
        instructions.pack(fill='both', expand=True, pady=10)
        
        inst_text = """Advanced Features Guide:

• Face Recognition: Register faces by pressing 'r' when face is detected, then recognize them
• Hand Gestures: Shows hand landmarks and basic gesture recognition (peace, thumbs up, etc.)
• Color Tracking: Tracks objects by color - select color with 'c' key
• Motion Detection: Highlights areas with movement
• QR/Barcode Scanner: Automatically detects and decodes QR codes and barcodes
• Distance Measurement: Click two points to measure distance (requires calibration)

General Controls:
• Press 'q' to exit camera mode
• Press 'h' for help overlay in camera window
• Right-click for context menu (in object detection mode)"""
        
        ttk.Label(instructions, text=inst_text, justify=tk.LEFT, font=("Arial", 9)).pack()
    
    def setup_settings_tab(self, parent):
        settings_frame = ttk.Frame(parent, padding="10")
        settings_frame.pack(expand=True, fill='both')
        
        # Title
        ttk.Label(settings_frame, text="Learning & Settings", 
                 font=("Arial", 16, "bold")).pack(pady=(0, 20))
        
        # Object Detection Learning Section
        learning_frame = ttk.LabelFrame(settings_frame, text="Object Detection Learning", padding="10")
        learning_frame.pack(fill='x', pady=10)
        
        ttk.Label(learning_frame, text="Improve object detection accuracy:", 
                 font=("Arial", 10, "bold")).pack(anchor='w')
        
        learn_text = """How to teach the AI:
1. Run Object Detection mode
2. When you see a wrong detection, right-click on it
3. Enter the correct label
4. The AI will learn from your corrections!"""
        
        ttk.Label(learning_frame, text=learn_text, justify=tk.LEFT).pack(anchor='w', pady=5)
        
        # Buttons for managing corrections
        btn_frame = ttk.Frame(learning_frame)
        btn_frame.pack(fill='x', pady=10)
        
        ttk.Button(btn_frame, text="View Corrections", 
                  command=self.show_corrections).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Clear All Corrections", 
                  command=self.clear_corrections).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Export Corrections", 
                  command=self.export_corrections).pack(side='left', padx=5)
        
        # Face Database Management
        face_frame = ttk.LabelFrame(settings_frame, text="Face Recognition Database", padding="10")
        face_frame.pack(fill='x', pady=10)
        
        ttk.Label(face_frame, text=f"Registered faces: {len(self.face_database)}", 
                 font=("Arial", 10)).pack(anchor='w')
        
        face_btn_frame = ttk.Frame(face_frame)
        face_btn_frame.pack(fill='x', pady=5)
        
        ttk.Button(face_btn_frame, text="Manage Face Database", 
                  command=self.manage_face_database).pack(side='left', padx=5)
        ttk.Button(face_btn_frame, text="Clear Face Database", 
                  command=self.clear_face_database).pack(side='left', padx=5)
        
        # Statistics
        stats_frame = ttk.LabelFrame(settings_frame, text="Usage Statistics", padding="10")
        stats_frame.pack(fill='x', pady=10)
        
        ttk.Label(stats_frame, text=f"Total corrections made: {len(self.user_corrections)}", 
                 font=("Arial", 10)).pack(anchor='w')
        ttk.Label(stats_frame, text="Most corrected objects: " + 
                 ", ".join(list(set(self.user_corrections.values()))[:3]), 
                 font=("Arial", 10)).pack(anchor='w')
    
    def set_mode(self, mode):
        if self.running:
            self.stop_camera()
        
        if mode == "stop":
            self.current_mode = "none"
            self.status_var.set("Camera stopped")
            return
            
        self.current_mode = mode
        self.status_var.set(f"Loading {mode.replace('_', ' ').title()} mode...")
        self.root.update()
        
        # Start camera in separate thread
        threading.Thread(target=self.start_camera, daemon=True).start()
    
    def load_models_for_mode(self, mode):
        """Lazy-load only the models needed for the current mode."""
        # 1) Object detection
        if mode == "object" and self.yolo_model is None:
            self.status_var.set("Loading YOLO model...")
            self.root.update()
            self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.yolo_model.conf = 0.4
            self.yolo_model.iou  = 0.45

        # 2) Emotion recognition (needs a face detector)
        if mode == "emotion" and self.face_cascade is None:
            self.status_var.set("Loading face detector for emotion...")
            self.root.update()
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

        # 3) Pose estimation (for pushup/pullup/squat and hand gestures)
        if mode in ["pushup", "pullup", "squat", "hand_gesture"] and self.pose is None:
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

        # 4) Hand detection (only for hand_gesture)
        if mode == "hand_gesture" and self.hands is None:
            self.status_var.set("Loading hand detection...")
            self.root.update()
            mp_hands = mp.solutions.hands
            self.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

    def start_camera(self):
        self.load_models_for_mode(self.current_mode)
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_var.set("Cannot open camera")
            return
        
        self.running = True
        self.status_var.set(f"Running {self.current_mode.replace('_', ' ').title()} mode - Press 'q' to stop")
        
        # Mouse callback for object detection corrections
        if self.current_mode == "object":
            cv2.namedWindow(f"CV Demo - {self.current_mode.title()} Mode")
            cv2.setMouseCallback(f"CV Demo - {self.current_mode.title()} Mode", self.mouse_callback)
        
        elif self.current_mode == "distance_measure":
            win_name = f"CV Demo - Distance Measure Mode"
            cv2.namedWindow(win_name)
            cv2.setMouseCallback(win_name, self.mouse_callback)
            # reset any old points
            self.measurement_points = []
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            self.frame_for_correction = frame.copy()
            
            # Process frame based on current mode
            if self.current_mode == "object":
                frame = self.process_object_detection(frame)
            elif self.current_mode == "emotion":
                frame = self.process_emotion_recognition(frame)
            elif self.current_mode in ["pushup", "pullup", "squat"]:
                frame = self.process_exercise_counter(frame)
            elif self.current_mode == "hand_gesture":
                frame = self.process_hand_gesture(frame)
            elif self.current_mode == "color_tracking":
                frame = self.process_color_tracking(frame)
            elif self.current_mode == "motion_detection":
                frame = self.process_motion_detection(frame)
            elif self.current_mode == "distance_measure":
                frame = self.process_distance_measurement(frame)
            
            # Add help overlay
            cv2.putText(frame, "Press 'h' for help, 'q' to quit", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(f"CV Demo - {self.current_mode.replace('_', ' ').title()} Mode", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                self.show_help_overlay()
        
        self.stop_camera()
    
    def process_object_detection(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.yolo_model(img_rgb, size=640)
        detections = results.xyxy[0]

        self.current_detections = []

        for *box, conf, cls in detections.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)
            original_label = self.yolo_model.names[class_id]

            # build a more unique key: classID_label_confidence%
            conf_pct = int(conf * 100)
            detection_key = f"{class_id}_{original_label}_{conf_pct}"

            if detection_key in self.user_corrections:
                label = self.user_corrections[detection_key]
                color = (0, 255, 255)  # Yellow = corrected
            else:
                label = original_label
                color = (0, 255, 0)    # Green = original

            self.current_detections.append({
                'box': (x1, y1, x2, y2),
                'label': label,
                'original_label': original_label,
                'confidence': conf,
                'detection_key': detection_key
            })

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            caption = f"{label} {conf:.2f}"
            cv2.putText(frame, caption, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # draw bottom instruction bar
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 30), (w, h), (50, 50, 50), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        instr = (
            "Right-click a box to correct its label  |  "
            f"Corrections loaded: {len(self.user_corrections)}"
        )
        cv2.putText(frame, instr, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return frame

    def process_face_recognition(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            x2, y2 = x + w, y + h
            face_img = frame[y:y2, x:x2]
            
            # Try to recognize face
            recognized_name = "Unknown"
            min_distance = float('inf')
            
            for name, stored_face in self.face_database.items():
                try:
                    result = DeepFace.verify(face_img, stored_face, enforce_detection=False)
                    distance = result['distance']
                    if distance < min_distance and distance < 0.6:  # Threshold
                        min_distance = distance
                        recognized_name = name
                except:
                    continue
            
            color = (0, 255, 0) if recognized_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            cv2.putText(frame, recognized_name, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.putText(frame, "Press 'r' to register new face", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame
    
    def process_hand_gesture(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return frame

        # landmark indices for tips and pips
        tips = [4, 8, 12, 16, 20]   # Thumb, Index, Middle, Ring, Pinky
        pips = [3, 6, 10, 14, 18]
        eps = 0.02

        for idx, (hand_landmarks, handedness) in enumerate(zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            )):
            # draw skeleton
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
            )

            lm = hand_landmarks.landmark
            # build a 5-element list: 1 = finger extended, 0 = folded
            fingers = []
            for tip_i, pip_i in zip(tips, pips):
                fingers.append(1 if lm[tip_i].y < lm[pip_i].y - eps else 0)

            total = sum(fingers)

            # gesture mapping
            if total == 0:
                gesture = "Fist"
            elif total == 5:
                gesture = "Open Hand"
            elif fingers == [1, 0, 0, 0, 0]:
                gesture = "Thumbs Up"
            elif fingers == [0, 1, 0, 0, 0]:
                gesture = "Pointing"
            elif fingers == [0, 0, 1, 0, 0]:
                gesture = "Middle Finger"
            elif fingers == [0, 0, 0, 1, 0]:
                gesture = "Ring Finger"
            elif fingers == [0, 0, 0, 0, 1]:
                gesture = "Pinky Finger"
            elif fingers == [0, 1, 1, 0, 0]:
                gesture = "Peace"
            else:
                gesture = f"{total} Fingers"

            # overlay handedness + gesture
            hand_label = handedness.classification[0].label  # "Left" or "Right"
            y = 50 + idx * 40
            cv2.putText(frame, f"{hand_label} Hand: {gesture}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    def process_color_tracking(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Track multiple colors
        for color_name, ranges in self.color_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            
            if len(ranges) == 4:  # Red has two ranges
                mask1 = cv2.inRange(hsv, ranges[0], ranges[1])
                mask2 = cv2.inRange(hsv, ranges[2], ranges[3])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv, ranges[0], ranges[1])
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Filter small objects
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{color_name.title()}: {area:.0f}px", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def process_motion_detection(self, frame):
        if not hasattr(self, 'prev_frame'):
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame
        
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_delta = cv2.absdiff(self.prev_frame, current_gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                motion_detected = True
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        status = "Motion Detected!" if motion_detected else "No Motion"
        color = (0, 255, 0) if motion_detected else (0, 0, 255)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        self.prev_frame = current_gray
        return frame
    
    def process_distance_measurement(self, frame):
        if not hasattr(self, 'measurement_points'):
            self.measurement_points = []
        
        # Draw existing points
        for i, point in enumerate(self.measurement_points):
            cv2.circle(frame, point, 5, (0, 255, 0), -1)
            cv2.putText(frame, f"P{i+1}", (point[0]+10, point[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw line and distance if we have 2 points
        if len(self.measurement_points) == 2:
            cv2.line(frame, self.measurement_points[0], self.measurement_points[1], (255, 0, 0), 2)
            
            # Calculate pixel distance
            pixel_distance = np.sqrt(
                (self.measurement_points[1][0] - self.measurement_points[0][0])**2 +
                (self.measurement_points[1][1] - self.measurement_points[0][1])**2
            )
            
            # Assume 1 pixel = 1mm (this needs calibration)
            real_distance = pixel_distance * 0.1  # Convert to cm
            
            mid_point = (
                (self.measurement_points[0][0] + self.measurement_points[1][0]) // 2,
                (self.measurement_points[0][1] + self.measurement_points[1][1]) // 2
            )
            
            cv2.putText(frame, f"{real_distance:.1f} cm", mid_point,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        cv2.putText(frame, "Click two points to measure distance", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame
    
    # Mouse callback for corrections and distance measurement
    def mouse_callback(self, event, x, y, flags, param):
        if self.current_mode == "object" and event == cv2.EVENT_RBUTTONDOWN:
            # Check if click is inside any detection box
            for detection in self.current_detections:
                x1, y1, x2, y2 = detection['box']
                if x1 <= x <= x2 and y1 <= y <= y2:
                    # Ask user for correction
                    self.correct_detection(detection)
                    break
        
        elif self.current_mode == "distance_measure" and event == cv2.EVENT_LBUTTONDOWN:
            if len(self.measurement_points) < 2:
                self.measurement_points.append((x, y))
            else:
                self.measurement_points = [(x, y)]
    
    def correct_detection(self, detection):
        # This would need to be called from the main thread
        def ask_correction():
            correct_label = simpledialog.askstring(
                "Correct Detection",
                f"Current: {detection['original_label']}\nEnter correct label:",
                parent=self.root
            )
            if correct_label and correct_label.strip():
                self.user_corrections[detection['detection_key']] = correct_label.strip()
                self.save_corrections()
                print(f"Learned: {detection['original_label']} -> {correct_label}")
        
        # Schedule this to run in the main thread
        self.root.after(0, ask_correction)
    
    # File I/O methods for learning
    def load_corrections(self):
        try:
            with open('user_corrections.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def save_corrections(self):
        with open('user_corrections.json', 'w') as f:
            json.dump(self.user_corrections, f, indent=2)
    
    def load_face_database(self):
        try:
            with open('face_database.pkl', 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {}
    
    def save_face_database(self):
        with open('face_database.pkl', 'wb') as f:
            pickle.dump(self.face_database, f)
    
    # UI Methods for settings
    def show_corrections(self):
        if not self.user_corrections:
            messagebox.showinfo("Corrections", "No corrections made yet!")
            return
        
        corrections_window = tk.Toplevel(self.root)
        corrections_window.title("User Corrections")
        corrections_window.geometry("500x400")
        
        text_widget = tk.Text(corrections_window, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(corrections_window, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        for original, corrected in self.user_corrections.items():
            text_widget.insert(tk.END, f"Original: {original}\nCorrected to: {corrected}\n\n")
        
        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        text_widget.config(state=tk.DISABLED)
    
    def clear_corrections(self):
        if messagebox.askyesno("Clear Corrections", "Are you sure you want to clear all corrections?"):
            self.user_corrections = {}
            self.save_corrections()
            messagebox.showinfo("Success", "All corrections cleared!")
    
    def export_corrections(self):
        if not self.user_corrections:
            messagebox.showinfo("Export", "No corrections to export!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"corrections_export_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.user_corrections, f, indent=2)
        
        messagebox.showinfo("Export Success", f"Corrections exported to {filename}")
    
    def register_face(self):
        if hasattr(self, 'current_faces') and self.current_faces:
            # Use the most recent face detection
            face_img = self.current_faces[-1]
            
            def ask_name():
                name = simpledialog.askstring("Register Face", "Enter person's name:", parent=self.root)
                if name and name.strip():
                    self.face_database[name.strip()] = face_img
                    self.save_face_database()
                    print(f"Registered face for: {name}")
            
            self.root.after(0, ask_name)
    
    def manage_face_database(self):
        if not self.face_database:
            messagebox.showinfo("Face Database", "No faces registered yet!")
            return
        
        db_window = tk.Toplevel(self.root)
        db_window.title("Face Database Manager")
        db_window.geometry("400x300")
        
        listbox = tk.Listbox(db_window)
        listbox.pack(fill="both", expand=True, padx=10, pady=10)
        
        for name in self.face_database.keys():
            listbox.insert(tk.END, name)
        
        def delete_selected():
            selection = listbox.curselection()
            if selection:
                name = listbox.get(selection[0])
                if messagebox.askyesno("Delete Face", f"Delete {name} from database?"):
                    del self.face_database[name]
                    self.save_face_database()
                    listbox.delete(selection[0])
        
        ttk.Button(db_window, text="Delete Selected", command=delete_selected).pack(pady=5)
    
    def clear_face_database(self):
        if messagebox.askyesno("Clear Database", "Are you sure you want to clear all registered faces?"):
            self.face_database = {}
            self.save_face_database()
            messagebox.showinfo("Success", "Face database cleared!")
    
    def show_help_overlay(self):
        help_text = {
            "object": "Right-click on wrong detections to correct them",
            "emotion": "Look at camera for emotion detection",
            "face_recognition": "Press 'r' to register a new face",
            "hand_gesture": "Show your hand to camera for gesture recognition",
            "color_tracking": "Objects of different colors will be tracked",
            "motion_detection": "Move in front of camera to detect motion",
            "qr_scanner": "Show QR codes or barcodes to camera",
            "distance_measure": "Click two points to measure distance",
            "pushup": "Do push-ups in front of camera",
            "pullup": "Do pull-ups in front of camera", 
            "squat": "Do squats in front of camera"
        }
        
        current_help = help_text.get(self.current_mode, "No specific help available")
        print(f"Help for {self.current_mode}: {current_help}")
    
    # Keep existing methods
    def process_emotion_recognition(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        self.current_faces = []  # Store for face registration
        
        for (x, y, w, h) in faces:
            x2, y2 = x + w, y + h
            face_img = frame[y:y2, x:x2]
            self.current_faces.append(face_img)
            
            try:
                # Use more robust emotion detection
                analysis = DeepFace.analyze(face_img, actions=["emotion"], enforce_detection=False)
                if isinstance(analysis, list):
                    analysis = analysis[0]
                dominant_emotion = analysis["dominant_emotion"]
                emotion_scores = analysis["emotion"]
                
                # Get confidence of dominant emotion
                confidence = emotion_scores[dominant_emotion]
                
            except Exception as e:
                dominant_emotion = "N/A"
                confidence = 0
                
            self.emotion_history.append(dominant_emotion)
            
            cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)
            
            if len(self.emotion_history) > 0:
                # Use weighted average of recent emotions
                emotion_counts = {}
                for emotion in self.emotion_history:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                common_emotion = max(emotion_counts, key=emotion_counts.get)
            else:
                common_emotion = dominant_emotion
                
            # Display emotion with confidence
            label = f"{common_emotion} ({confidence:.1f}%)" if confidence > 0 else common_emotion
            cv2.putText(frame, label, (x, y - 10),
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
                    
                    # Add form feedback
                    if elbow_angle < 70:
                        form_feedback = "Good depth!"
                        feedback_color = (0, 255, 0)
                    elif elbow_angle < 90:
                        form_feedback = "Go lower"
                        feedback_color = (0, 255, 255)
                    else:
                        form_feedback = "Starting position"
                        feedback_color = (255, 255, 255)
                    
                    cv2.putText(frame, f"Push-ups: {count} | Angle: {elbow_angle:.1f}°", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(frame, form_feedback, (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, feedback_color, 2)
                               
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
                
                # Add form feedback for squats
                if knee_angle < 90:
                    form_feedback = "Great depth!"
                    feedback_color = (0, 255, 0)
                elif knee_angle < 120:
                    form_feedback = "Good form"
                    feedback_color = (0, 255, 255)
                else:
                    form_feedback = "Stand up straight"
                    feedback_color = (255, 255, 255)
                
                cv2.putText(frame, f"Squats: {count} | Angle: {knee_angle:.1f}°",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                cv2.putText(frame, form_feedback, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, feedback_color, 2)
        else:
            cv2.putText(frame, "Stand in view of camera", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
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
        self.angle_history = deque(maxlen=5)  # For smoothing

    def update(self, angle):
        self.angle_history.append(angle)
        # Use average of recent angles for stability
        avg_angle = sum(self.angle_history) / len(self.angle_history)
        
        if self.state == "UP" and avg_angle < self.down_thresh:
            self.state = "DOWN"
        elif self.state == "DOWN" and avg_angle > self.up_thresh:
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