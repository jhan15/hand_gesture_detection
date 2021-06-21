import argparse
import cv2
import numpy as np
import time
from osascript import osascript

from hand import HandDetector
from utils.utils import draw_vol_bar, draw_landmarks, two_landmark_distance
from utils.utils import update_trajectory, check_trajectory


CAM_W = 1280                                # camera width
CAM_H = 800                                 # camera height
TEXT_COLOR = (102,51,0)                     # text color
LINE_COLOR_HIGH = (0,0,255)                 # landmark color high
LINE_COLOR_LOW = (0,255,0)                  # landmark color low
VOL_RANGE = [0, 100]                        # system volume range
BAR_X_RANGE = [1000, 1200]                  # bar x position range
LEN_RANGE = [20, 150]                       # range of thumb and index fingertips
STEP_THRESHOLD = [30, 130]                  # threshold of step control


def vol_control(control='none',             # gesture control type
                max_hands=2,                # maximum number of hands detected
                detect_conf=0.8,            # detection confidence level
                track_conf=0.5,             # tracking confidence level
                step=10,                    # step control size
                trajectory_size=12,         # trajectory size
                ):
    
    cap = cv2.VideoCapture(0)
    cap.set(3, CAM_W)
    cap.set(4, CAM_H)
    detector = HandDetector(max_num_hands=max_hands,
                            min_detection_confidence=detect_conf,
                            min_tracking_confidence=track_conf)

    vol = (VOL_RANGE[0] + VOL_RANGE[1]) // 2
    vol_bar = (BAR_X_RANGE[0] + BAR_X_RANGE[1]) // 2
    osascript("set volume output volume {}".format(vol))

    ptime = 0
    ctime = 0
    window_name = 'Volume control'
    trajectory = list()

    while True:
        _, img = cap.read()
        img = cv2.flip(img, 1)
        img, hands = detector.detect_hands(img)

        # control
        if hands:
            # none control mode
            if control == 'none':
                img = detector.draw_landmarks(img)
            
            # continuous control mode
            if control == 'pinch_conti':
                landmarks = hands[-1]['lm'] # use the firstly detected hand
                length, pt1, pt2 = two_landmark_distance(landmarks, 4, 8)
                draw_landmarks(img, pt1, pt2)
                vol = np.interp(length, LEN_RANGE, VOL_RANGE)
                vol_bar = np.interp(length, LEN_RANGE, BAR_X_RANGE)
                osascript("set volume output volume {}".format(vol))

            # step control mode
            if control == 'pinch_step':
                landmarks = hands[-1]['lm']
                length, pt1, pt2 = two_landmark_distance(landmarks, 4, 8)

                if length > STEP_THRESHOLD[1]:
                    draw_landmarks(img, pt1, pt2, LINE_COLOR_HIGH)
                elif length < STEP_THRESHOLD[0]:
                    draw_landmarks(img, pt1, pt2, LINE_COLOR_LOW)
                else:
                    draw_landmarks(img, pt1, pt2)

                trajectory = update_trajectory(length, trajectory, trajectory_size)
                up = False
                down = False

                if len(trajectory) == trajectory_size and length > STEP_THRESHOLD[1]:
                    up = check_trajectory(trajectory, direction=1)
                    if up:
                        vol = min(vol + step, VOL_RANGE[1])
                        osascript("set volume output volume {}".format(vol))
                
                if len(trajectory) == trajectory_size and length < STEP_THRESHOLD[0]:
                    down = check_trajectory(trajectory, direction=-1)
                    if down:
                        vol = max(vol - step, VOL_RANGE[0])
                        osascript("set volume output volume {}".format(vol))
                
                if up or down:
                    vol_bar = np.interp(vol, VOL_RANGE, BAR_X_RANGE)
                    trajectory = []           
             
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        
        draw_vol_bar(img, vol_bar, vol, BAR_X_RANGE)

        cv2.putText(img, f'FPS: {int(fps)}', (50,38), 0, 0.8, TEXT_COLOR, 2, lineType=cv2.LINE_AA)

        cv2.imshow(window_name, img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', type=str, default='none')
    parser.add_argument('--max_hands', type=int, default=2)
    parser.add_argument('--detect_conf', type=float, default=0.7)
    parser.add_argument('--track_conf', type=float, default=0.5)
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--trajectory_size', type=int, default=12)
    opt = parser.parse_args()

    vol_control(**vars(opt))
    