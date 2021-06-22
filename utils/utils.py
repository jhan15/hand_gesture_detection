import cv2
import math
import numpy as np


TEXT_COLOR = (102,51,0)       # text color
BAR_COLOR = (76,0,153)        # bar color
LINE_COLOR = (255,255,255)    # landmark connection color
LM_COLOR = (255,51,255)       # landmark color
BOX_COLOR = (153,0,153)


def find_boundary_lm(landmarks):
    xs = np.array(landmarks)[:,0]
    ys = np.array(landmarks)[:,1]
    lm_x_max, lm_x_min = np.argmax(xs), np.argmin(xs)
    lm_y_max, lm_y_min = np.argmax(ys), np.argmin(ys)

    return [lm_x_max, lm_x_min, lm_y_max, lm_y_min]


def get_finger_state(acc_angle, threshold):
    finger_state = None
    if acc_angle > threshold[1]:
        finger_state = 0
    elif acc_angle < threshold[0]:
        finger_state = 2
    else:
        finger_state = 1
    
    return finger_state


def check_hand_direction(landmarks):
    direction = None
    mcp_joints = [5, 9, 13, 17]
    wrist = landmarks[0]
    mcp_x = np.array([landmarks[i][0] for i in mcp_joints])
    mcp_y = np.array([landmarks[i][1] for i in mcp_joints])
    
    if np.all(mcp_x > wrist[0]):
        direction = 'right'
    if np.all(mcp_x < wrist[0]):
        direction = 'left'
    if np.all(mcp_y > wrist[1]):
        direction = 'down'
    if np.all(mcp_y < wrist[1]):
        direction = 'up'
    
    return direction


def draw_bounding_box(landmarks, detected_gesture, img, tor=40):
    xs = np.array(landmarks)[:,0]
    ys = np.array(landmarks)[:,1]
    x_max, x_min = np.max(xs), np.min(xs)
    y_max, y_min = np.max(ys), np.min(ys)
    cv2.rectangle(img, (x_min-tor,y_min-tor), (x_max+tor,y_max+tor), BOX_COLOR, 2, lineType=cv2.LINE_AA)
    cv2.rectangle(img, (x_min-tor,y_min-2*tor), (x_max+tor,y_min-tor), BOX_COLOR, -1, lineType=cv2.LINE_AA)
    cv2.putText(img, f'{detected_gesture}', (x_min-tor+5,y_min-tor-10), 0, 1, LINE_COLOR, 3, lineType=cv2.LINE_AA)


def map_gesture(finger_states, direction, boundary, gestures):
    detected_gesture = None
    for ges, temp in gestures.items():
        count = 0
        if temp['finger states'] == finger_states:
            count += 1
        if temp['direction'] == direction:
            count += 1
        if temp['boundary'] is None:
            count += 1
        else:
            flag = 0
            for bound, lm in temp['boundary'].items():
                if boundary[bound] != lm:
                    flag = 1
                    break
            if flag == 0:
                count += 1
        if count == 3:
            detected_gesture = ges
            break
    
    return detected_gesture

def calculate_angle(joint1, joint2, joint3):
    vec21 = np.array(joint1) - np.array(joint2)
    vec23 = np.array(joint3) - np.array(joint2)
    cosine_angle = np.dot(vec21, vec23) / (np.linalg.norm(vec21) * np.linalg.norm(vec23))
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
