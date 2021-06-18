import argparse
import cv2
import numpy as np
import time
from osascript import osascript

from utils.hand import HandDetector
from utils.utils import draw_vol_bar, draw_landmarks, two_landmark_distance
from utils.utils import update_buffer, check_buffer


cam_w = 640                 # camera width
cam_h = 480                 # camera height
text_color = (255,0,0)      # text color
lm_color_up = (0,0,255)     # landmark color high
lm_color_down = (0,255,0)   # landmark color low
vol_range = [0, 100]        # system volume range
bar_x_range = [350, 550]    # bar x position range
len_range = [20, 150]       # length range of thumb tip and index finge
step_threshold = [30, 130]  # threshold of step control     


def vol_control(control='pinch_conti',      # gesture control type
                max_hands=2,                # maximum number of hands detected
                detect_conf=0.7,            # detection confidence level
                track_conf=0.5,             # tracking confidence level
                step=10,                    # step control size
                buffer=5,                   # buffer size
                ):
    
    cap = cv2.VideoCapture(1)
    cap.set(3, cam_w)
    cap.set(4, cam_h)
    detector = HandDetector(max_num_hands=max_hands,
                            min_detection_confidence=detect_conf,
                            min_tracking_confidence=track_conf)

    vol = (vol_range[0] + vol_range[1]) // 2
    vol_bar = (bar_x_range[0] + bar_x_range[1]) // 2
    osascript("set volume output volume {}".format(vol))

    ptime = 0
    ctime = 0
    window_name = 'Volume control'
    buffer_list = list()

    while True:
        _, img = cap.read()
        img = cv2.flip(img, 1)
        img, hands = detector.detect_hands(img)

        # control
        if hands:
            if control == 'none':
                img = detector.draw_landmarks(img)
            
            if control == 'pinch_conti':
                landmarks = hands[-1]['lm'] # use the firstly detected hand
                length, pt1, pt2 = two_landmark_distance(landmarks, 4, 8)
                draw_landmarks(img, pt1, pt2)
                vol = np.interp(length, len_range, vol_range)
                vol_bar = np.interp(length, len_range, bar_x_range)

            if control == 'pinch_step':
                landmarks = hands[-1]['lm']
                length, pt1, pt2 = two_landmark_distance(landmarks, 4, 8)

                if length > step_threshold[1]:
                    draw_landmarks(img, pt1, pt2, lm_color_up)
                elif length < step_threshold[0]:
                    draw_landmarks(img, pt1, pt2, lm_color_down)
                else:
                    draw_landmarks(img, pt1, pt2)

                buffer_list = update_buffer(length, buffer_list, buffer)
                up = False
                down = False

                if len(buffer_list) == buffer and length > step_threshold[1]:
                    up = check_buffer(buffer_list, direction=1)
                    if up:
                        vol = min(vol + step, vol_range[1])
                
                if len(buffer_list) == buffer and length < step_threshold[0]:
                    down = check_buffer(buffer_list, direction=-1)
                    if down:
                        vol = max(vol - step, vol_range[0])
                
                if up or down:
                    vol_bar = np.interp(vol, vol_range, bar_x_range)
                    buffer_list = []

            osascript("set volume output volume {}".format(vol))
             
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        
        draw_vol_bar(img, vol_bar, vol, bar_x_range)

        cv2.putText(img, f'FPS: {int(fps)}', (30,40), 0, 0.8, text_color, 2, lineType=cv2.LINE_AA)

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
    parser.add_argument('--buffer', type=int, default=5)
    opt = parser.parse_args()

    vol_control(**vars(opt))
    