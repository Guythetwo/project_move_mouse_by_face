import cv2
from catboost import CatBoostRegressor
import mediapipe as mp
import numpy as np
from collections import deque
import pyautogui
import math
from keyboard import is_pressed
from collections import deque

# --- Model Loading ---
file_model_x = "model/catboost_regressor_model_X.cbm"
file_model_y = "model/catboost_regressor_model_Y.cbm"

try:
    model_x = CatBoostRegressor()
    model_x.load_model(file_model_x)
    model_y = CatBoostRegressor()
    model_y.load_model(file_model_y)
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please ensure the model files are in the 'model' directory.")
    exit()

# --- Constants and Initialization ---

# Face landmarks for head pose
LANDMARKS = {
    "left": 234,
    "right": 454,
    "top": 10,
    "bottom": 152,
    "front": 1,
}

# Eye landmarks
LEFT_IRIS = [473] # Note: In MediaPipe's new model, left/right might be swapped depending on camera mirroring. 473 is often the user's right eye (left side of the screen).
RIGHT_IRIS = [468] # User's left eye.
LEFT_EYE = [362, 263]
RIGHT_EYE = [33, 133]

# Screen resolution
MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()
CENTER_X = MONITOR_WIDTH // 2
CENTER_Y = MONITOR_HEIGHT // 2

# Filter for smoothing mouse movement
filter_length = 8
ray_origins = deque(maxlen=filter_length)
ray_directions = deque(maxlen=filter_length)

smoothed_x = 1
smoothed_y = -1
alpha = 0.4 # ปรับค่านี้เพื่อเปลี่ยนความสมูท

# MediaPipe Face Mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Helper Functions ---

def get_center(landmarks, indices, image_shape):
    """Calculates the average center point of a list of landmarks."""
    h, w = image_shape
    pts = [ (landmarks[idx].x * w, landmarks[idx].y * h) for idx in indices ]
    pts = np.array(pts)
    return np.mean(pts, axis=0)  # Returns (x, y)

def landmark_to_np(landmark, w, h):
    """Converts a MediaPipe landmark to a NumPy array with scaling."""
    return np.array([landmark.x * w, landmark.y * h, landmark.z * w])

def calc_yaw_pitch(iris_center, eye_center):
    """Calculates a numerical representation of yaw/pitch from iris deviation."""
    dx = iris_center[0] - eye_center[0]
    dy = eye_center[1] - iris_center[1] # Y is inverted in image coordinates
    yaw = np.degrees(np.arctan2(dx, eye_center[0]))
    pitch = np.degrees(np.arctan2(dy, eye_center[1]))
    return yaw, pitch

# --- Main Program ---

# Move mouse to center of the screen to start
pyautogui.moveTo(CENTER_X, CENTER_Y)

# Start webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

while cap.isOpened():
    # Initialize data list for each frame
    data = []
    
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally for a mirror-like view
    #frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark

        # --- Eye Gaze Visualization (Optional, but helpful for debugging) ---
        left_eye_center = get_center(face_landmarks, LEFT_EYE, (h, w))
        left_iris_center = get_center(face_landmarks, LEFT_IRIS, (h, w))
        left_yaw, left_pitch = calc_yaw_pitch(left_iris_center, left_eye_center)

        right_eye_center = get_center(face_landmarks, RIGHT_EYE, (h, w))
        right_iris_center = get_center(face_landmarks, RIGHT_IRIS, (h, w))
        right_yaw, right_pitch = calc_yaw_pitch(right_iris_center, right_eye_center)

        # Draw points on eyes
        cv2.circle(frame, tuple(np.int32(left_eye_center)), 3, (255, 0, 0), -1)
        cv2.circle(frame, tuple(np.int32(left_iris_center)), 3, (0, 255, 0), -1)
        cv2.circle(frame, tuple(np.int32(right_eye_center)), 3, (255, 0, 0), -1)
        cv2.circle(frame, tuple(np.int32(right_iris_center)), 3, (0, 255, 0), -1)

        # Display eye gaze values
        cv2.putText(frame, f"L yaw: {left_yaw:.2f} pitch: {left_pitch:.2f}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"R yaw: {right_yaw:.2f} pitch: {right_pitch:.2f}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # --- Head Pose Calculation ---
        key_points = {}
        for name, idx in LANDMARKS.items():
            pt = landmark_to_np(face_landmarks[idx], w, h)
            key_points[name] = pt
            x, y = int(pt[0]), int(pt[1])
            data.extend([x, y]) # Add landmark x,y to our feature list

        # Calculate orientation vectors
        right_axis = (key_points["right"] - key_points["left"])
        right_axis /= np.linalg.norm(right_axis)

        up_axis = (key_points["top"] - key_points["bottom"])
        up_axis /= np.linalg.norm(up_axis)

        forward_axis = np.cross(right_axis, up_axis)
        forward_axis /= np.linalg.norm(forward_axis)
        forward_axis = -forward_axis  # Flip Z direction

        # --- Smoothing ---
        ray_origins.append(np.mean([pt for pt in key_points.values()], axis=0))
        ray_directions.append(forward_axis)
        
        avg_origin = np.mean(ray_origins, axis=0)
        avg_direction = np.mean(ray_directions, axis=0)
        avg_direction /= np.linalg.norm(avg_direction)

        # --- Angle Calculation ---
        reference_forward = np.array([0, 0, -1])

        # Yaw (Horizontal)
        yaw_rad = math.atan2(avg_direction[0], avg_direction[2]) # More stable calculation
        yaw_deg = np.degrees(yaw_rad)

        # Pitch (Vertical)
        pitch_rad = math.asin(-avg_direction[1])
        pitch_deg = np.degrees(pitch_rad)

        # Append head pose angles to our feature list
        if len(data) == 10:
            data.append(yaw_deg)
            data.append(pitch_deg)

        # --- Prediction and Mouse Control ---
        if len(data) == 12:
            input_data = np.array(data).reshape(1, -1) # Reshape for single prediction
            
            # Predict screen coordinates
            predicted_screen_x = model_x.predict(input_data)
            predicted_screen_y = model_y.predict(input_data)

            if smoothed_x == 1:
                smoothed_x = predicted_screen_x
                smoothed_y = predicted_screen_y
            else:
                # คำนวณค่า smoothed ใหม่
                smoothed_x = alpha * predicted_screen_x + (1 - alpha) * smoothed_x
                smoothed_y = alpha * predicted_screen_y + (1 - alpha) * smoothed_y

            pyautogui.moveTo(smoothed_x, smoothed_y)

        # --- Visualization of head pose vector ---
        ray_length = 150
        ray_end = avg_origin - avg_direction * ray_length
        cv2.line(frame, 
                 (int(avg_origin[0]), int(avg_origin[1])), 
                 (int(ray_end[0]), int(ray_end[1])), 
                 (15, 255, 0), 3)

    # Display the resulting frame
    cv2.imshow("Eye and Head Tracker", frame)

    # Exit condition: Press 'a' or 'A'
    if is_pressed('a') or is_pressed('A') or cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()