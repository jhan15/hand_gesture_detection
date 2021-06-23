import cv2
import time
import numpy as np

from hand import HandDetector
from utils.templates import Gesture
from utils.utils import calculate_angle, get_thumb_state, get_finger_state, map_gesture
from utils.utils import draw_bounding_box


THUMB_THRESH = 5.5
THUMB_MCP_THRESH = 2.7
NON_THUMB_THRESH = [8.4, 8.1, 6.9, 6]

THUMB_STATES ={
    0: 'straight',
    1: 'bent',
    2: 'closed'
}
NON_THUMB_STATES = {
    0: 'straight',
    1: 'claw',
    2: 'bent',
    3: 'closed',
    4: 'clenched'
}

CAM_W = 1280
CAM_H = 800
TEXT_COLOR = (102,51,0)


class GestureDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2,
                min_detection_confidence=0.8, min_tracking_confidence=0.5):
        
        self.hand_detector = HandDetector(static_image_mode,
                                          max_num_hands,
                                          min_detection_confidence,
                                          min_tracking_confidence)
    
    
    def check_finger_states(self, landmarks, img):
        finger_states = [None] * 5
        joint_angles = np.zeros((5,3))
        
        for i in range(5):
            joints = [0, 4*i+1, 4*i+2, 4*i+3, 4*i+4]
            joint_angles[i] = np.array(
                [calculate_angle(landmarks[joints[j]],
                                 landmarks[joints[j+1]],
                                 landmarks[joints[j+2]]) for j in range(3)]
            )
            if i == 0:
                acc_angle = joint_angles[i, 1:].sum()
                finger_states[i] = get_thumb_state(acc_angle,
                                                   joint_angles[i, 1],
                                                   THUMB_THRESH,
                                                   THUMB_MCP_THRESH)
            else:
                acc_angle = joint_angles[i].sum()
                finger_states[i] = get_finger_state(acc_angle, NON_THUMB_THRESH)

            # pt = landmarks[joints[4]]
            # cv2.putText(img, f'{round(acc_angle,2)}', (pt[0]+5,pt[1]+5), 0, 0.5, (0,255,255), 2)
            # pt = landmarks[joints[3]]
            # cv2.putText(img, f'{round(joint_angles[i, 2],2)}', (pt[0]+5,pt[1]+5), 0, 0.5, (0,255,0), 2)
            # pt = landmarks[joints[2]]
            # cv2.putText(img, f'{round(joint_angles[i, 1],2)}', (pt[0]+5,pt[1]+5), 0, 0.5, (255,0,0), 2)
            # pt = landmarks[joints[1]]
            # cv2.putText(img, f'{round(joint_angles[i, 0],2)}', (pt[0]+5,pt[1]+5), 0, 0.5, TEXT_COLOR, 2)

            # pt = landmarks[joints[4]]
            # cv2.putText(img, f'{finger_states[i]}', (pt[0]+5,pt[1]+5), 0, 0.5, (0,255,255), 2)
        
        return finger_states
    
    def detect_gestures(self, img):
        hands = self.hand_detector.detect_hands(img)
        self.hand_detector.draw_landmarks(img)
        detected_gesture = None

        if hands:
            hand = hands[-1]
            ges = Gesture(hand['label'])
            finger_states = self.check_finger_states(hand['lm'], img)
            detected_gesture = map_gesture(finger_states,
                                           hand['direction'],
                                           hand['boundary'],
                                           ges.gestures)
            if detected_gesture:
                draw_bounding_box(hand['lm'], detected_gesture, img)

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
    