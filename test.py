import mediapipe as mp
from catboost import CatBoostClassifier
import cv2
import pyautogui

filename = "model/CatBoostClassifier_model.cbm"
model = CatBoostClassifier()
model.load_model(filename)
important_mouth_indices = [61, 291, 13, 14, 78, 308]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            data = []
            for idx in important_mouth_indices:
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    data.extend([landmark.x, landmark.y, landmark.z])
                    
            predicted = model.predict(data)
            cv2.putText(frame, f"yawn : {predicted}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Face Tracker", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()