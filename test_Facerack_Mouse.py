import cv2
from catboost import CatBoostRegressor, CatBoostClassifier
import mediapipe as mp
import numpy as np
from collections import deque
import pyautogui
import math
from keyboard import is_pressed

file_model_x = "model/catboost_regressor_model_X.cbm"
file_model_y = "model/catboost_regressor_model_Y.cbm"
filename = "model/CatBoostClassifier_model.cbm"

model_x = CatBoostRegressor()
model_x.load_model(file_model_x)
model_y = CatBoostRegressor()
model_y.load_model(file_model_y)
model_cilck = CatBoostClassifier()
model_cilck.load_model(filename)


# -MASK Face-
LANDMARKS = {
    "left": 234,
    "right": 454,   
    "top": 10,
    "bottom": 152,
    "front": 1,
}

important_mouth_indices = [61, 291, 13, 14, 78, 308]

LEFT_IRIS = [468] 
RIGHT_IRIS = [473]
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]

# Screen resolution
MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()
CENTER_X = MONITOR_WIDTH // 2
CENTER_Y = MONITOR_HEIGHT // 2

# Filter length for smoothing
filter_length = 8
ray_origins = deque(maxlen=filter_length)
ray_directions = deque(maxlen=filter_length)

predicted = 0

center_position = []

# MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_center(landmarks, indices, image_shape):
    h, w = image_shape
    pts = []
    for idx in indices:
        lm = landmarks[idx]
        pts.append((lm.x * w, lm.y * h))
    pts = np.array(pts)
    return np.mean(pts, axis=0)  # x, y

def get_center_for_fix():
    pass

# คำนวณ yaw/pitch จากตาดำเทียบกับจุดกลางตา
def calc_yaw_pitch(iris_center, eye_center):
    dx = iris_center[0] - eye_center[0]  # แนวนอน
    dy = eye_center[1] - iris_center[1]  # แนวตั้ง (กลับทิศ)
    yaw = np.degrees(np.arctan2(dx, eye_center[0]))   # ซ้าย/ขวา
    pitch = np.degrees(np.arctan2(dy, eye_center[1])) # ขึ้น/ลง
    return yaw, pitch

pyautogui.moveTo(CENTER_X,CENTER_Y)

# Webcam input
cap = cv2.VideoCapture(0)

def landmark_to_np(landmark, w, h):
    return np.array([landmark.x * w, landmark.y * h, landmark.z * w])

def get_3d_coordinates(landmark, image_width, image_height):
    return np.array([landmark.x * image_width, landmark.y * image_height, landmark.z])

def move_mouse_face(array):
    pass

while cap.isOpened():
    data = []
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # ดวงตาซ้าย
            data_click = []
            left_eye_center = get_center(face_landmarks.landmark, LEFT_EYE, (h, w))
            left_iris_center = get_center(face_landmarks.landmark, LEFT_IRIS, (h, w))
            left_yaw, left_pitch = calc_yaw_pitch(left_iris_center, left_eye_center)

            # ดวงตาขวา
            right_eye_center = get_center(face_landmarks.landmark, RIGHT_EYE, (h, w))
            right_iris_center = get_center(face_landmarks.landmark, RIGHT_IRIS, (h, w))
            right_yaw, right_pitch = calc_yaw_pitch(right_iris_center, right_eye_center)

            # วาดจุด
            cv2.circle(frame, tuple(np.int32(left_eye_center)), 3, (255, 0, 0), -1)
            cv2.circle(frame, tuple(np.int32(left_iris_center)), 3, (0, 255, 0), -1)
            cv2.circle(frame, tuple(np.int32(right_eye_center)), 3, (255, 0, 0), -1)
            cv2.circle(frame, tuple(np.int32(right_iris_center)), 3, (0, 255, 0), -1)

            # แสดงผล
            cv2.putText(frame, f"L yaw: {left_yaw:.2f} pitch: {left_pitch:.2f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"R yaw: {right_yaw:.2f} pitch: {right_pitch:.2f}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
            for idx in important_mouth_indices:
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
                    data_click.extend([landmark.x, landmark.y, landmark.z])

        cv2.circle(frame, (w // 2, h // 2), radius=10, color=(0, 0, 255), thickness=-1)
        face_landmarks = results.multi_face_landmarks[0].landmark

        key_points = {}
        for name, idx in LANDMARKS.items():
            pt = landmark_to_np(face_landmarks[idx], w, h)
            key_points[name] = pt
            x, y = int(pt[0]), int(pt[1])
            data.extend([x, y]) # Use extend for clarity

        left = key_points["left"]
        right = key_points["right"]
        top = key_points["top"]
        bottom = key_points["bottom"]
        
        # Calculate orientation vectors
        right_axis = (right - left)
        right_axis /= np.linalg.norm(right_axis)

        up_axis = (top - bottom)
        up_axis /= np.linalg.norm(up_axis)

        forward_axis = np.cross(right_axis, up_axis)
        forward_axis /= np.linalg.norm(forward_axis)
        forward_axis = -forward_axis  # Flip direction

        center = np.mean([key_points[name] for name in LANDMARKS], axis=0)

        # Append current values to buffers for smoothing
        ray_origins.append(center)
        ray_directions.append(forward_axis)

        # Calculate smoothed (average) origin and direction
        avg_origin = np.mean(ray_origins, axis=0)
        avg_direction = np.mean(ray_directions, axis=0)
        avg_direction /= np.linalg.norm(avg_direction)

        reference_forward = np.array([0, 0, -1])  # Z-axis into the screen

        # Horizontal (yaw) angle from reference (project onto XZ plane)
        xz_proj = np.array([avg_direction[0], 0, avg_direction[2]])
        xz_proj /= np.linalg.norm(xz_proj)

        yaw_rad = math.acos(np.clip(np.dot(reference_forward, xz_proj), -1.0, 1.0))
        if avg_direction[0] < 0:
            yaw_rad = -yaw_rad  # left is negative

        # Vertical (pitch) angle from reference (project onto YZ plane)
        yz_proj = np.array([0, avg_direction[1], avg_direction[2]])
        yz_proj /= np.linalg.norm(yz_proj)
        pitch_rad = math.acos(np.clip(np.dot(reference_forward, yz_proj), -1.0, 1.0))
        if avg_direction[1] > 0:
            pitch_rad = -pitch_rad  # up is positive

        #Specify 

        # Convert to degrees and re-center around 0
        yaw_deg = np.degrees(yaw_rad)
        pitch_deg = np.degrees(pitch_rad)

        #this results in the center being 180, +10 left = -170, +10 right = +170

        #convert left rotations to 0-180
        if yaw_deg < 0:
            yaw_deg = abs(yaw_deg)
        elif yaw_deg < 180:
            yaw_deg = 360 - yaw_deg

        if pitch_deg < 0:
            pitch_deg = 360 + pitch_deg

        raw_yaw_deg = yaw_deg
        raw_pitch_deg = pitch_deg
        # Draw the gaze direction ray
        ray_length = 150 # Adjusted for better visualization
        ray_end = avg_origin - avg_direction * ray_length
        cv2.line(frame, 
                 (int(avg_origin[0]), int(avg_origin[1])), 
                 (int(ray_end[0]), int(ray_end[1])), 
                 (15, 255, 0), 3)

    cv2.imshow("Face Tracker", frame)

    if len(data) == 10:
        data.append(raw_yaw_deg)
        data.append(raw_pitch_deg)

    new_data = np.array(data)

    if len(data) == 12:
        predicted_screen_x = model_x.predict(new_data)
        predicted_screen_y = model_y.predict(new_data)

        pyautogui.moveTo(predicted_screen_x,predicted_screen_y)
    
    if len(data_click) == 18:
        predicted = model_cilck.predict(data_click)
        cv2.putText(frame, f"yawn : {predicted}", (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if predicted == 1:
            pyautogui.click()

    # --- This is the crucial part that was missing/changed ---
    if is_pressed('a') or is_pressed('A'):
        break

cap.release()
cv2.destroyAllWindows()
# The line 'filename.close()' was removed as it's incorrect.

cap.release()
cv2.destroyAllWindows()
# The line 'filename.close()' was removed as it's incorrect.