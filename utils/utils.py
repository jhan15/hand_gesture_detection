import cv2
import math
import numpy as np


TEXT_COLOR = (102,51,0)       # text color
BAR_COLOR = (76,0,153)        # bar color
LINE_COLOR = (255,255,255)    # landmark connection color
LM_COLOR = (255,51,255)       # landmark color


def draw_bounding_box(landmarks, detected_gesture, img, tor=40):
    xs = np.array(landmarks)[:,0]
    ys = np.array(landmarks)[:,1]
    x_max, x_min = np.max(xs), np.min(xs)
    y_max, y_min = np.max(ys), np.min(ys)
    cv2.rectangle(img, (x_min-tor,y_min-tor), (x_max+tor,y_max+tor), BAR_COLOR, 2, lineType=cv2.LINE_AA)
    cv2.rectangle(img, (x_min-tor,y_min-2*tor), (x_max+tor,y_min-tor), BAR_COLOR, -1, lineType=cv2.LINE_AA)
    cv2.putText(img, f'{detected_gesture}', (x_min-tor+5,y_min-tor-5), 0, 1.2, LINE_COLOR, 3, lineType=cv2.LINE_AA)


def map_gesture(finger_states, gestures):
    detected_gesture = None
    if finger_states in gestures.values():
        for ges, state in gestures.items():
            if state == finger_states:
                detected_gesture = ges
                break
    
    return detected_gesture

def calculate_angle(pt1, pt2, pt3):
    pt21 = np.array(pt1) - np.array(pt2)
    pt23 = np.array(pt3) - np.array(pt2)
    cosine_angle = np.dot(pt21, pt23) / (np.linalg.norm(pt21) * np.linalg.norm(pt23))
    angle = np.arccos(cosine_angle)

    return angle


def draw_vol_bar(img, vol_bar, vol, bar_x_range):
    cv2.rectangle(img, (bar_x_range[0],20), (bar_x_range[1],40), BAR_COLOR, 1, lineType=cv2.LINE_AA)
    cv2.rectangle(img, (bar_x_range[0],20), (int(vol_bar),40), BAR_COLOR, -1, lineType=cv2.LINE_AA)
    cv2.putText(img, f'{int(vol)}', (bar_x_range[1]+10,38), 0, 0.8, TEXT_COLOR, 2, lineType=cv2.LINE_AA)


def draw_landmarks(img, pt1, pt2, color=LINE_COLOR):
    cv2.circle(img, pt1, 10, LM_COLOR, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, pt2, 10, LM_COLOR, -1, lineType=cv2.LINE_AA)
    cv2.line(img, pt1, pt2, color, 3)


def two_landmark_distance(landmarks, id1, id2):
    x1, y1 = landmarks[id1][0], landmarks[id1][1]
    x2, y2 = landmarks[id2][0], landmarks[id2][1]
    length = math.hypot(x2 - x1, y2 - y1)
    
    return length, (x1,y1), (x2,y2)


def update_trajectory(length, trajectory, trajectory_size):
    if len(trajectory) < trajectory_size:
        trajectory.append(length)
    else:
        trajectory.pop(0)
        trajectory.append(length)
    
    return trajectory


def check_trajectory(trajectory, direction):
    if direction == 1:
        return all(i < j for i, j in zip(trajectory, trajectory[1:]))
    if direction == -1:
        return all(i > j for i, j in zip(trajectory, trajectory[1:]))
