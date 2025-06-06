import cv2
import torch
import numpy as np
import mediapipe as mp
from deepface import DeepFace
from collections import deque

# ───────────────────────────────────────────────────────────────────────
# 1) SET UP OBJECT DETECTION (YOLOv5 via Torch Hub)
# ───────────────────────────────────────────────────────────────────────
# This will download (once) and load a small YOLOv5 model. You can swap 
# 'yolov5s' for 'yolov5m' or others if you have GPU/need more accuracy.
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.4  # confidence threshold (0.0–1.0)
model.iou = 0.45  # NMS IoU threshold

# ───────────────────────────────────────────────────────────────────────
# 2) SET UP FACIAL EMOTION RECOGNITION (DeepFace)
# ───────────────────────────────────────────────────────────────────────
# We'll call DeepFace.analyze(...) on each detected face. That function 
# returns a dict containing 'dominant_emotion'. This is slower than a 
# hand-rolled CNN, but DeepFace works out of the box.
#  
# NOTE: DeepFace will automatically download a face-analysis model (~20MB) 
# the first time it runs.

# You can also supply a face detector (e.g., 'opencv', 'mtcnn', 'dlib') in 
# DeepFace.analyze(...) to speed things up or reduce false positives. We'll 
# let it do the default for simplicity.


# ───────────────────────────────────────────────────────────────────────
# 3) SET UP POSE DETECTION FOR EXERCISE COUNTERS (MediaPipe)
# ───────────────────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Utility functions to compute angles between three points
def calculate_angle(a, b, c):
    """
    Computes the angle at point b formed by the points a‐b‐c (in degrees).
    a, b, c must be NumPy arrays: [x, y].
    """
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint
    c = np.array(c)  # Endpoint

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


# ───────────────────────────────────────────────────────────────────────
# 4) EXERCISE COUNTERS: push‐up, pull‐up, squat
#    We keep a small state machine per exercise:
#    - When the relevant joint angle goes below a “down” threshold, we mark as DOWN.
#    - When the angle then goes above an “up” threshold, we count one rep and go back to UP.
# ───────────────────────────────────────────────────────────────────────
class RepCounter:
    def __init__(self, down_threshold, up_threshold):
        self.down_thresh = down_threshold
        self.up_thresh = up_threshold
        self.state = "UP"  # can be "UP" or "DOWN"
        self.count = 0

    def update(self, angle):
        if self.state == "UP" and angle < self.down_thresh:
            self.state = "DOWN"
        elif self.state == "DOWN" and angle > self.up_thresh:
            self.state = "UP"
            self.count += 1
        return self.count


# Instantiate counters with heuristic angle thresholds.
# – Push-up: elbow angle ∈ [160°, 180°] when fully extended (UP); < 90° when DOWN
pushup_counter = RepCounter(down_threshold=90, up_threshold=160)

# – Pull-up: elbow angle ∈ [160°, 180°] when hanging (UP); < 90° at chin‐over‐bar (DOWN)
pullup_counter = RepCounter(down_threshold=90, up_threshold=160)

# – Squat: knee angle ∈ [160°, 180°] when standing (UP); < 90° when in squat (DOWN)
squat_counter = RepCounter(down_threshold=90, up_threshold=160)


# ───────────────────────────────────────────────────────────────────────
# 5) MAIN LOOP: capture video → run detections → draw overlays → show frame
# ───────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    # To smooth emotion recognition (avoid flicker), keep last N predictions in a deque.
    emotion_history = deque(maxlen=5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally for mirror view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # ============================================
        # A) OBJECT DETECTION (YOLOv5)
        # ============================================
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img_rgb, size=640)  # run inference
        detections = results.xyxy[0]  # tensor of detections: [x1, y1, x2, y2, conf, cls]

        for *box, conf, cls in detections.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)
            label = model.names[class_id]
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Label + confidence
            caption = f"{label} {conf:.2f}"
            cv2.putText(
                frame, caption, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        # ============================================
        # B) FACIAL EMOTION RECOGNITION (DeepFace)
        # ============================================
        # We first detect faces with OpenCV’s built-in Haar cascades (fast), then crop
        # and pass to DeepFace for emotion. Alternatively, DeepFace can detect faces itself.
        # Here we'll use OpenCV’s frontal face cascade for speed.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # For each face, analyze emotion
        for (x, y, w_f, h_f) in faces:
            x2f, y2f = x + w_f, y + h_f
            face_img = frame[y:y2f, x:x2f]
            try:
                analysis = DeepFace.analyze(face_img, actions=["emotion"], enforce_detection=False)
                dominant_emotion = analysis["dominant_emotion"]
            except Exception as e:
                dominant_emotion = "N/A"
            emotion_history.append(dominant_emotion)

            # Draw face bounding box and label
            cv2.rectangle(frame, (x, y), (x2f, y2f), (255, 0, 0), 2)
            # Show the most common emotion in recent frames to smooth
            if len(emotion_history) > 0:
                common_emotion = max(set(emotion_history), key=emotion_history.count)
            else:
                common_emotion = dominant_emotion
            cv2.putText(
                frame, common_emotion, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
            )

        # ============================================
        # C) POSE ESTIMATION (MediaPipe) & EXERCISE COUNT
        # ============================================
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(img_rgb)

        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark
            # Convert normalized landmarks to pixel coordinates
            def landmark_to_point(idx):
                lm = landmarks[idx]
                return np.array([int(lm.x * w), int(lm.y * h)])

            # Draw skeleton
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                results_pose.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

            # --- PUSH-UP COUNTER (use elbow angle) ---
            # We'll use right side for simplicity: shoulder = 12, elbow = 14, wrist = 16
            shoulder = landmark_to_point(12)
            elbow = landmark_to_point(14)
            wrist = landmark_to_point(16)
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            pu_count = pushup_counter.update(elbow_angle)
            cv2.putText(
                frame, f"Push-ups: {pu_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
            )

            # --- PULL-UP COUNTER (also use elbow angle, but assume user is facing camera) ---
            # In a realistic scenario you'd check if their body is vertical & detect bar.
            # Here we reuse the same angle logic for demonstration.
            pl_count = pullup_counter.update(elbow_angle)
            cv2.putText(
                frame, f"Pull-ups: {pl_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2
            )

            # --- SQUAT COUNTER (use knee angle) ---
            # Right hip = 24, right knee = 26, right ankle = 28
            hip = landmark_to_point(24)
            knee = landmark_to_point(26)
            ankle = landmark_to_point(28)
            knee_angle = calculate_angle(hip, knee, ankle)
            sq_count = squat_counter.update(knee_angle)
            cv2.putText(
                frame, f"Squats: {sq_count}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2
            )

        # ============================================
        # FINAL: SHOW THE FRAME
        # ============================================
        cv2.imshow("Ultimate Computer Vision Demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
