import cv2
import mediapipe as mp
import numpy as np


class MediaPipeFaceModel:
    def __init__(self, **kwargs):
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.2
        )

    def predict(self, *data):
        img_bgr = data[0]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_bgr.shape[:2]
        results = self.face_mesh.process(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            return []
        outs = []
        for face_landmarks in results.multi_face_landmarks:
            landmarks = []
            for landmark in face_landmarks.landmark:
                landmarks.append([landmark.x * w, landmark.y * h])
            outs.append(np.array(landmarks))
        return outs
