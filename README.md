```markdown
# Advanced Computer Vision Suite

An all-in-one desktop application offering multiple computer vision demos and tools, built with Python and Tkinter.

## Table of Contents
1. [Features](#features)
2. [Technologies & Libraries](#technologies--libraries)
3. [How It Works](#how-it-works)
4. [Future Improvements](#future-improvements)

## Features
- **Smart Object Detection**  
  - YOLOv5-based real-time detection with right-click correction and persistent learning.  
- **Emotion Recognition**  
  - Haar cascade + DeepFace analysis for dominant emotion with smoothing over recent frames.  
- **Exercise Counters**  
  - Push-up, pull-up, and squat counters using MediaPipe pose landmarks, angle calculation, and form feedback.  
- **Hand Gesture Detection**  
  - MediaPipe Hands-based recognition of fist, open hand, thumbs up, pointing, middle/ring/pinky finger, peace sign, and numeric finger counts, with left/right labeling.  
- **Color Object Tracking**  
  - HSV-based multi-color tracking (red, blue, green, yellow, orange, purple, cyan, magenta).  
- **Motion Detection**  
  - Frame-difference motion highlighting.  
- **Distance Measurement**  
  - User clicks two points to measure pixel distance (converted to cm).  

## Technologies & Libraries
- **GUI**: Tkinter, ttk  
- **Vision & ML**:  
  - OpenCV (cv2)  
  - PyTorch (YOLOv5 via `torch.hub`)  
  - MediaPipe (Pose & Hands)  
  - DeepFace (emotion analysis)  
- **Utilities**:  
  - NumPy, collections.deque, PIL / Pillow  
  - Threading for non-blocking camera loop  
- **Persistence**: JSON for corrections, pickle for face database  

## How It Works
1. **Lazy Loading**  
   - Models and cascades are only initialized when a mode is first activated, keeping startup fast.  
2. **Camera Loop**  
   - Each mode’s processing function is called on every frame, then displayed via OpenCV’s `imshow`.  
3. **User Learning**  
   - Object Detection: right-click triggers a Tkinter prompt; new labels persist across sessions.  
4. **Gesture & Pose Logic**  
   - Uses MediaPipe landmarks to compute whether each finger is up/down and angle-based rep counting for exercises.  

## Future Improvements
- Add support for webcam selection & resolution settings  
- Integrate a calibration step for distance measurement units  
- Export session logs & statistics (e.g., total exercise reps)  
- Modularize processing pipelines into separate files/packages  
```
