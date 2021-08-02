"""
Control volume by hand gestures.

Usage:
    $ python3 vol_controller.py --control continuous
"""

import argparse
import cv2
import numpy as np
import time
from osascript import osascript

from gesture import GestureDetector
from utils.utils import two_landmark_distance, draw_vol_bar, draw_landmarks
from utils.utils import update_trajectory, check_trajectory


CAM_W = 1280
CAM_H = 720
TEXT_COLOR = (102,51,0)
ACTI_COLOR = (0,255,0)
VOL_RANGE = [0, 100]
BAR_X_RANGE = [50, CAM_W//5]


def vol_control(control='continuous', vol_step=10, traj_size=10):
    cap = cv2.VideoCapture(0)
    cap.set(3, CAM_W)
    cap.set(4, CAM_H)
    ges_detector = GestureDetector(max_num_hands=1)

    vol = (VOL_RANGE[0] + VOL_RANGE[1]) // 2
    vol_bar = (BAR_X_RANGE[0] + BAR_X_RANGE[1]) // 2
    osascript("set volume output volume {}".format(vol))

    ptime = 0
    ctime = 0
    trajectory = list()
    target_gestures = ['Pinch', 'C shape']
    wrist, thumb_tip, index_tip = 0, 4, 8
    activated = False
    len_range = None

    while True:
        _, img = cap.read()
        img = cv2.flip(img, 1)
        gesture = ges_detector.detect_gesture(img, 'single')
        hands = ges_detector.hand_detector.decoded_hands

        if gesture:
            hand = hands[-1]
            landmarks = hand['landmarks']
            if gesture in target_gestures:
                ges_detector.draw_gesture_box(img)
            if gesture == target_gestures[0]:
                if not activated:
                    base_len = two_landmark_distance(landmarks[wrist], landmarks[thumb_tip])
                    len_range = [0.1*base_len, 0.6*base_len]
                    step_threshold = [0.2*base_len, 0.9*base_len]
                activated = True
            if activated and gesture == target_gestures[1]:
                activated = False
        
        if activated:
            if hands:
                hand = hands[-1]
                landmarks = hand['landmarks']
                pt1 = landmarks[thumb_tip][:2]
                pt2 = landmarks[index_tip][:2]
                length = two_landmark_distance(pt1, pt2)
                
                # continuous control mode
                if control == 'continuous':
                    draw_landmarks(img, pt1, pt2)
                    finger_states = ges_detector.check_finger_states(hand)
                    if finger_states[4] > 2:
                        vol = np.interp(length, len_range, VOL_RANGE)
                        vol_bar = np.interp(length, len_range, BAR_X_RANGE)
                        osascript("set volume output volume {}".format(vol))

                # step control mode
                if control == 'step':
                    draw_landmarks(img, pt1, pt2)
                    trajectory = update_trajectory(length, trajectory, traj_size)
                    up = False
                    down = False
                    if len(trajectory) == traj_size and length > step_threshold[1]:
                        up = check_trajectory(trajectory, direction=1)
                        if up:
                            vol = min(vol + vol_step, VOL_RANGE[1])
                            osascript("set volume output volume {}".format(vol))
                    if len(trajectory) == traj_size and length < step_threshold[0]:
                        down = check_trajectory(trajectory, direction=-1)
                        if down:
                            vol = max(vol - vol_step, VOL_RANGE[0])
                            osascript("set volume output volume {}".format(vol))
                    if up or down:
                        vol_bar = np.interp(vol, VOL_RANGE, BAR_X_RANGE)
                        trajectory = []           
             
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        
        pt1 = (30,20)
        pt2 = (BAR_X_RANGE[1]+80,150)
        draw_vol_bar(img, pt1, pt2, vol_bar, vol, fps, BAR_X_RANGE, activated)

        cv2.imshow('Volume controller', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', type=str, default='continuous',
                        help='volume control mode (default: continuous)')
    parser.add_argument('--vol_step', type=int, default=10,
                        help='volume update step for step control (default: 10)')
    parser.add_argument('--traj_size', type=int, default=10,
                        help='trajetory size (default: 10)')
    opt = parser.parse_args()

    vol_control(**vars(opt))
