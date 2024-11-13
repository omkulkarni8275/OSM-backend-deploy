import cv2
import mediapipe as mp
import base64
import numpy as np
import time
from playsound import playsound
import os
from collections import deque
from plyer import notification
import threading

last_sound_time = 0
sound_interval = 40

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

previous_blink_state = False
blink_count = 0
blink_history = []
blink_smooth_window = deque(maxlen=5)

is_calibrated = False
calibration_frames = 0
calibration_shoulder_angles = []
calibration_neck_angles = []
shoulder_threshold = None
neck_threshold = None
posture_smooth_window = deque(maxlen=5)
last_alert_time = time.time()
alert_cooldown = 10

sound_file = "music.mp3"

brightness_threshold = 300
low_light_notification_sent = False
last_notification_time = time.time()
notification_interval = 300

calibration_frames_target = 70

def play_sound_in_thread(sound_file):
    try:
        if os.path.exists(sound_file):
            threading.Thread(target=playsound, args=(sound_file,), daemon=True).start()
        else:
            print(f"Sound file {sound_file} not found.")
    except Exception as e:
        print(f"Error playing sound: {e}")

def check_lighting_condition(frame):
    global low_light_notification_sent, last_notification_time
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    if brightness < brightness_threshold:
        current_time = time.time()
        if not low_light_notification_sent or (current_time - last_notification_time >= notification_interval):
            notification.notify(
                title="Lighting Condition Alert",
                message="Inadequate lighting detected. Please increase lighting for accurate detection.",
                app_name="Posture and Blink Detection",
                timeout=10
            )
            last_notification_time = current_time
            low_light_notification_sent = True
        return False
    low_light_notification_sent = False
    return True

def calculate_angle(point1, point2, point3):
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def draw_angle(frame, point1, point2, point3, angle, color):
    cv2.putText(frame, str(int(angle)), 
                tuple(np.add(point2, (10, -10)).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

def analyze_posture(frame, landmarks):
    global is_calibrated, calibration_frames, shoulder_threshold, neck_threshold, last_alert_time, last_sound_time

    left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1]),
                     int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0]))
    right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1]),
                      int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0]))
    left_ear = (int(landmarks[mp_pose.PoseLandmark.LEFT_EAR].x * frame.shape[1]),
                int(landmarks[mp_pose.PoseLandmark.LEFT_EAR].y * frame.shape[0]))

    shoulder_angle = calculate_angle(left_shoulder, right_shoulder, (right_shoulder[0], right_shoulder[1] - 100))
    neck_angle = calculate_angle(left_ear, left_shoulder, (left_shoulder[0], left_shoulder[1] - 100))

    if not is_calibrated and calibration_frames < calibration_frames_target:
        calibration_shoulder_angles.append(shoulder_angle)
        calibration_neck_angles.append(neck_angle)
        calibration_frames += 1
        return f"Calibrating... {calibration_frames}/{calibration_frames_target}", (255, 255, 0)
    
    if not is_calibrated:
        shoulder_threshold = np.mean(calibration_shoulder_angles) - 5
        neck_threshold = np.mean(calibration_neck_angles) - 5
        is_calibrated = True
        print(f"Calibration complete. Shoulder threshold: {shoulder_threshold:.1f}, Neck threshold: {neck_threshold:.1f}")

    posture_smooth_window.append((shoulder_angle, neck_angle))
    if len(posture_smooth_window) > 10:
        posture_smooth_window.pop(0)

    smooth_shoulder_angle = np.mean([angle[0] for angle in posture_smooth_window])
    smooth_neck_angle = np.mean([angle[1] for angle in posture_smooth_window])

    current_time = time.time()

    if smooth_shoulder_angle < shoulder_threshold or smooth_neck_angle < neck_threshold:
        if current_time - last_alert_time > alert_cooldown:
            print("Poor posture detected! Please sit up straight.")
            if current_time - last_sound_time >= sound_interval:
                last_sound_time = current_time
                play_sound_in_thread(sound_file)
        return "Poor Posture", (0, 0, 255)
    else:
        return "Good Posture", (0, 255, 0)

def analyze_focus(frame):
    global previous_blink_state, blink_count, blink_history
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            left_eye_top = face_landmarks.landmark[159]
            left_eye_bottom = face_landmarks.landmark[145]
            right_eye_top = face_landmarks.landmark[386]
            right_eye_bottom = face_landmarks.landmark[374]

            left_eye_distance = abs(left_eye_top.y - left_eye_bottom.y)
            right_eye_distance = abs(right_eye_top.y - right_eye_bottom.y)
            eye_open_threshold = 0.02

            avg_eye_distance = (left_eye_distance + right_eye_distance) / 2

            blink_smooth_window.append(avg_eye_distance)
            if len(blink_smooth_window) > 5:
                blink_smooth_window.pop(0)
            smooth_eye_distance = np.mean(blink_smooth_window)

            if smooth_eye_distance < eye_open_threshold:
                if not previous_blink_state:
                    blink_count += 1
                    previous_blink_state = True
                    blink_history.append(time.time())
            else:
                previous_blink_state = False

    current_time = time.time()
    blink_history = [t for t in blink_history if current_time - t < 60]

    return blink_count

def generate_frame():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    global blink_count
    blink_count = 0
    posture_status = "Unknown"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        check_lighting_condition(frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            posture_status, posture_color = analyze_posture(frame, landmarks)
            blink_count = analyze_focus(frame)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame, posture_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, posture_color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Blinks: {blink_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_encoded = base64.b64encode(buffer).decode('utf-8')

        yield frame_encoded, posture_status, blink_count
    
    cap.release()
