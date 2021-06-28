import cv2
import mediapipe as mp
import time
import numpy as np

from utils.utils import check_hand_direction, find_boundary_lm


CAM_W = 1280
CAM_H = 800
TEXT_COLOR = (102, 51, 0)


class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2,
                min_detection_confidence=0.8, min_tracking_confidence=0.5):
        
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
                                    self.static_image_mode,
                                    self.max_num_hands,
                                    self.min_detection_confidence,
                                    self.min_tracking_confidence)
    
    def detect_hands(self, img):
        decoded_hands = None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        if self.results.multi_hand_landmarks:
            h, w, _ = img.shape
            num_hands = len(self.results.multi_hand_landmarks)
            decoded_hands = [None] * num_hands

            for i in range(num_hands):
                decoded_hands[i] = dict()
                handedness = self.results.multi_handedness[i]
                hand_landmarks = self.results.multi_hand_landmarks[i]

                decoded_hands[i]['index'] = handedness.classification[0].index
                decoded_hands[i]['label'] = handedness.classification[0].label

                lm_list = list()
                wrist_z = hand_landmarks.landmark[0].z

                for lm in hand_landmarks.landmark:
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)
                    cz = int((lm.z - wrist_z) * w)
                    lm_list.append([cx, cy, cz])
                
                lm_array = np.array(lm_list)
                decoded_hands[i]['lm'] = lm_array
                decoded_hands[i]['direction'], decoded_hands[i]['facing'] = check_hand_direction(lm_array, decoded_hands[i]['label'])
                decoded_hands[i]['boundary'] = find_boundary_lm(lm_array)
        
        return decoded_hands
    
    def draw_landmarks(self, img):
        if self.results.multi_hand_landmarks:
            for landmarks in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(img, landmarks, self.mp_hands.HAND_CONNECTIONS)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, CAM_W)
    cap.set(4, CAM_H)
    detector = HandDetector()
    ptime = 0
    ctime = 0

    while True:
        _, img = cap.read()
        img = cv2.flip(img, 1)
        detector.detect_hands(img)
        detector.draw_landmarks(img)
        
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, f'FPS: {int(fps)}', (30,40), 0, 0.8, TEXT_COLOR , 2)

        cv2.imshow('Hand detection', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
