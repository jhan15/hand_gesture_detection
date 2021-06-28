import cv2
import numpy as np


TEXT_COLOR = (102,51,0)
BAR_COLOR = (76,0,153)
LINE_COLOR = (255,255,255)
LM_COLOR = (255,51,255)
BOX_COLOR = (153,0,153)


def find_boundary_lm(landmarks):
    """ Get the landmarks/joints with maximum x, minimum x, maximum y, and minimum y values. """
    xs = landmarks[:,0]
    ys = landmarks[:,1]
    lm_x_max, lm_x_min = np.argmax(xs), np.argmin(xs)
    lm_y_max, lm_y_min = np.argmax(ys), np.argmin(ys)

    return [lm_x_max, lm_x_min, lm_y_max, lm_y_min]


def get_finger_state(joint_angles, threshold, compa_len_thres=None, compa_len=None):
    """ Define a finger's state by it's joint angles. """
    acc_angle = joint_angles.sum()
    finger_state = None
    
    new_threshold = threshold.copy()
    new_threshold.append(-np.inf)
    new_threshold.insert(0, np.inf)
    
    for i in range(len(new_threshold)-1):
        if new_threshold[i] > acc_angle >= new_threshold[i+1]:
            finger_state = i
            break
    
    if compa_len:
        if finger_state == 0 and compa_len < compa_len_thres:
            finger_state = 1
    
    return finger_state


def check_hand_direction(landmarks, label):
    """ Check hand's direction. """
    direction = None
    facing = None
    mcp_joints = [1, 5, 9, 13, 17]
    wrist = landmarks[0]

    finger_mcp_x = np.mean(landmarks[mcp_joints[1:], 0])
    finger_mcp_y = np.mean(landmarks[mcp_joints[1:], 1])

    finger_wrist_x = np.absolute(finger_mcp_x - wrist[0])
    finger_wrist_y = np.absolute(finger_mcp_y - wrist[1])

    if finger_wrist_x > finger_wrist_y:
        if finger_mcp_x < wrist[0]:
            direction = 'left'
            if label == 'Left':
                facing = 'front' if landmarks[mcp_joints[0]][1] < landmarks[mcp_joints[4]][1] else 'back'
            else:
                facing = 'front' if landmarks[mcp_joints[0]][1] > landmarks[mcp_joints[4]][1] else 'back'
        else:
            direction = 'right'
            if label == 'Left':
                facing = 'front' if landmarks[mcp_joints[0]][1] > landmarks[mcp_joints[4]][1] else 'back'
            else:
                facing = 'front' if landmarks[mcp_joints[0]][1] < landmarks[mcp_joints[4]][1] else 'back'
    else:
        if finger_mcp_y < wrist[1]:
            direction = 'up'
            if label == 'Left':
                facing = 'front' if landmarks[mcp_joints[0]][0] > landmarks[mcp_joints[4]][0] else 'back'
            else:
                facing = 'front' if landmarks[mcp_joints[0]][0] < landmarks[mcp_joints[4]][0] else 'back'
        else:
            direction = 'down'
            if label == 'Left':
                facing = 'front' if landmarks[mcp_joints[0]][0] < landmarks[mcp_joints[4]][0] else 'back'
            else:
                facing = 'front' if landmarks[mcp_joints[0]][0] > landmarks[mcp_joints[4]][0] else 'back'
    
    return direction, facing


def draw_bounding_box(landmarks, detected_gesture, img, tor=40):
    """ Draw a bounding box of detected hand with gesture label. """
    xs = landmarks[:,0]
    ys = landmarks[:,1]
    x_max, x_min = np.max(xs), np.min(xs)
    y_max, y_min = np.max(ys), np.min(ys)
    cv2.rectangle(img, (x_min-tor,y_min-tor), (x_max+tor,y_max+tor),
                                BOX_COLOR, 2, lineType=cv2.LINE_AA)
    cv2.rectangle(img, (x_min-tor,y_min-2*tor), (x_max+tor,y_min-tor),
                                BOX_COLOR, -1, lineType=cv2.LINE_AA)
    cv2.putText(img, f'{detected_gesture}', (x_min-tor+5,y_min-tor-10), 0, 1,
                                LINE_COLOR, 3, lineType=cv2.LINE_AA)


def map_gesture(finger_states, direction, boundary, gestures, spec=4):
    """ Map detected gesture fetures to a pre-defined gesture template. """
    detected_gesture = None
    for ges, temp in gestures.items():
        count = 0
        
        # check finger states
        if spec in temp['finger states']:
            if temp['finger states'] == finger_states:
                count += 1
        else:
            new_finger_states = [x if x!=spec else (spec-1) for x in finger_states]
            if temp['finger states'] == new_finger_states:
                count += 1
        
        # check direction
        if temp['direction'] == direction:
            count += 1
        
        # check boundary
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


def calculate_angle(joints):
    """ Calculate the angle of three points. """
    vec1 = joints[0][:2] - joints[1][:2]
    vec2 = joints[2][:2] - joints[1][:2]
    cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(cosine_angle)

    return angle


def calculate_thumb_angle(joints, label, facing):
    vec1 = joints[0][:2] - joints[1][:2]
    vec2 = joints[2][:2] - joints[1][:2]

    if label == 'Left':
        cross = np.cross(vec1, vec2) if facing == 'front' else np.cross(vec2, vec1)
    else:
        cross = np.cross(vec2, vec1) if facing == 'front' else np.cross(vec1, vec2)
    dot = np.dot(vec1, vec2)
    angle = np.arctan2(cross, dot)
    if angle < 0:
        angle += 2 * np.pi
    
    return angle


def two_landmark_distance(vec1, vec2, dim=3):
    """ Calculate the distance between two landmarks. """
    vec = vec2[:dim] - vec1[:dim]
    distance = np.linalg.norm(vec)
    
    return distance


#########################################################################
# below functions are specifically for volume control, need check later #
#########################################################################

def draw_vol_bar(img, vol_bar, vol, bar_x_range):
    """ Draw a volume bar. """
    cv2.rectangle(img, (bar_x_range[0],20), (bar_x_range[1],40),
                                    BAR_COLOR, 1, lineType=cv2.LINE_AA)
    cv2.rectangle(img, (bar_x_range[0],20), (int(vol_bar),40),
                                    BAR_COLOR, -1, lineType=cv2.LINE_AA)
    cv2.putText(img, f'{int(vol)}', (bar_x_range[1]+10,38), 0, 0.8,
                                    TEXT_COLOR, 2, lineType=cv2.LINE_AA)


def draw_landmarks(img, pt1, pt2, color=LINE_COLOR):
    """ Draw two landmarks and the connection line. """
    cv2.circle(img, pt1, 10, LM_COLOR, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, pt2, 10, LM_COLOR, -1, lineType=cv2.LINE_AA)
    cv2.line(img, pt1, pt2, color, 3)


def update_trajectory(length, trajectory, trajectory_size):
    """ Update the trajectory list. """
    if len(trajectory) < trajectory_size:
        trajectory.append(length)
    else:
        trajectory.pop(0)
        trajectory.append(length)
    
    return trajectory


def check_trajectory(trajectory, direction):
    """ Check whether the trajectory is always increasing or decreasing or not. """
    if direction == 1:
        return all(i < j for i, j in zip(trajectory, trajectory[1:]))
    if direction == -1:
        return all(i > j for i, j in zip(trajectory, trajectory[1:]))
