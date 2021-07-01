"""
Detect hand gestures on streams.

Usage:
    $ python3 gesture.py --mode 'single'
"""

import argparse
import cv2
import time
import numpy as np

from hand import HandDetector
from utils.templates import Gesture
from utils.utils import two_landmark_distance
from utils.utils import calculate_angle, calculate_thumb_angle, get_finger_state
from utils.utils import map_gesture, draw_bounding_box, draw_fingertips


THUMB_THRESH = [9, 8]
NON_THUMB_THRESH = [8.6, 7.8, 6.6, 6.1]

BENT_RATIO_THRESH = [0.76, 0.88, 0.85, 0.65]

CAM_W = 640
CAM_H = 480
TEXT_COLOR = (102,51,0)


# A hand gesture detector to detect different gestures according to pre-defined gesture templates.
class GestureDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2,
                 min_detection_confidence=0.8, min_tracking_confidence=0.5):
        
        self.hand_detector = HandDetector(static_image_mode,
                                          max_num_hands,
                                          min_detection_confidence,
                                          min_tracking_confidence)
    
    def check_finger_states(self, hand):
        landmarks = hand['landmarks']
        label = hand['label']
        facing = hand['facing']

        finger_states = [None] * 5
        joint_angles = np.zeros((5,3)) # 5 fingers and 3 angles each

        # wrist to index finger mcp
        d1 = two_landmark_distance(landmarks[0], landmarks[5])
        
        # loop for 5 fingers
        for i in range(5):
            joints = [0, 4*i+1, 4*i+2, 4*i+3, 4*i+4]
            if i == 0:
                joint_angles[i] = np.array(
                    [calculate_thumb_angle(landmarks[joints[j:j+3]], label, facing) for j in range(3)]
                )
                finger_states[i] = get_finger_state(joint_angles[i], THUMB_THRESH)
            else:
                joint_angles[i] = np.array(
                    [calculate_angle(landmarks[joints[j:j+3]]) for j in range(3)]
                )
                d2 = two_landmark_distance(landmarks[joints[1]], landmarks[joints[4]])
                finger_states[i] = get_finger_state(joint_angles[i], NON_THUMB_THRESH)
                
                if finger_states[i] == 0 and d2/d1 < BENT_RATIO_THRESH[i-1]:
                    finger_states[i] = 1
        
        return finger_states
    
    def detect_gestures(self, img, mode):
        hands = self.hand_detector.detect_hands(img)
        self.hand_detector.draw_landmarks(img)
        detected_gesture = None

        if hands:
            if mode == 'single':
                hand = hands[-1]
                ges = Gesture(hand['label'])
                finger_states = self.check_finger_states(hand)
                draw_fingertips(hand['landmarks'], finger_states, img)
                
                detected_gesture = map_gesture(finger_states,
                                            hand['direction'],
                                            hand['boundary'],
                                            ges.gestures)
                
                if detected_gesture:
                    draw_bounding_box(hand['landmarks'], detected_gesture, img)
            # if mode == 'double' and len(hands) == 2:

        return detected_gesture


def main(mode='single'):
    cap = cv2.VideoCapture(0)
    cap.set(3, CAM_W)
    cap.set(4, CAM_H)
    ges_detector = GestureDetector()
    ptime = 0
    ctime = 0

    while True:
        _, img = cap.read()
        img = cv2.flip(img, 1)
        ges_detector.detect_gestures(img, mode)
        
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='single',
                        help='single/double-hand gestures (default: single)')
    opt = parser.parse_args()

    main(**vars(opt))
    