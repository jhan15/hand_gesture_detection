import cv2
import time

from hand import HandDetector
from utils.templates import Gesture
from utils.utils import map_gesture, draw_bounding_box


CAM_W = 1280
CAM_H = 800
TEXT_COLOR = (102,51,0)


class GestureDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2,
                min_detection_confidence=0.8, min_tracking_confidence=0.5):
        
        self.ges = Gesture()
        self.hand_detector = HandDetector(static_image_mode,
                                          max_num_hands,
                                          min_detection_confidence,
                                          min_tracking_confidence)
    
    def detect_gestures(self, img):
        hands = self.hand_detector.detect_hands(img)
        self.hand_detector.draw_landmarks(img)
        detected_gesture = None

        if hands:
            landmarks = hands[-1]['lm']
            finger_states = self.hand_detector.check_finger_states(landmarks, img)
            detected_gesture = map_gesture(finger_states, self.ges.gestures)
            if detected_gesture:
                draw_bounding_box(landmarks, detected_gesture, img)

        return detected_gesture


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, CAM_W)
    cap.set(4, CAM_H)
    ges_detector = GestureDetector()
    ptime = 0
    ctime = 0

    while True:
        _, img = cap.read()
        img = cv2.flip(img, 1)
        ges_detector.detect_gestures(img)
        
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, f'FPS: {int(fps)}', (50,38), 0, 0.8, TEXT_COLOR, 2, lineType=cv2.LINE_AA)

        cv2.imshow('Gesture detection', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
    