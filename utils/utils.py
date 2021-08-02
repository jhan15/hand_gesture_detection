import cv2
import numpy as np


TEXT_COLOR = (243,236,27)
BAR_COLOR = (51,255,51)
LINE_COLOR = (255,255,255)
LM_COLOR = (255,51,255)
LABEL_COLOR = (102,51,0)

THUMB_STATES ={
    0: ['straight', (121,49,255)],
    1: ['bent', (243,166,56)],
    2: ['closed', (107,29,92)]
}
NON_THUMB_STATES = {
    0: ['straight', (121,49,255)],
    1: ['claw', (76,166,255)],
    2: ['bent', (243,166,56)],
    3: ['closed', (178,30,180)],
    4: ['clenched', (107,29,92)]
}


def find_boundary_lm(landmarks):
    """ Get the landmarks with maximum x, minimum x, maximum y, and minimum y values. """
    xs = landmarks[:,0]
    ys = landmarks[:,1]
    lm_x_max, lm_x_min = np.argmax(xs), np.argmin(xs)
    lm_y_max, lm_y_min = np.argmax(ys), np.argmin(ys)

    return [lm_x_max, lm_x_min, lm_y_max, lm_y_min]


def check_hand_direction(landmarks, label):
    """ Check hand's direction and facing. """
    direction = None
    facing = None
    mcp_joints = [5, 9, 13, 17]
    wrist = landmarks[0]
    thumb_mcp = landmarks[1]
    pinky_mcp = landmarks[17]

    mcp_x_avg = np.mean(landmarks[mcp_joints, 0])
    mcp_y_avg = np.mean(landmarks[mcp_joints, 1])

    mcp_wrist_x = np.absolute(mcp_x_avg - wrist[0])
    mcp_wrist_y = np.absolute(mcp_y_avg - wrist[1])

    if mcp_wrist_x > mcp_wrist_y:
        if mcp_x_avg < wrist[0]:
            direction = 'left'
            if label == 'left':
                facing = 'front' if thumb_mcp[1] < pinky_mcp[1] else 'back'
            else:
                facing = 'front' if thumb_mcp[1] > pinky_mcp[1] else 'back'
        else:
            direction = 'right'
            if label == 'left':
                facing = 'front' if thumb_mcp[1] > pinky_mcp[1] else 'back'
            else:
                facing = 'front' if thumb_mcp[1] < pinky_mcp[1] else 'back'
    else:
        if mcp_y_avg < wrist[1]:
            direction = 'up'
            if label == 'left':
                facing = 'front' if thumb_mcp[0] > pinky_mcp[0] else 'back'
            else:
                facing = 'front' if thumb_mcp[0] < pinky_mcp[0] else 'back'
        else:
            direction = 'down'
            if label == 'left':
                facing = 'front' if thumb_mcp[0] < pinky_mcp[0] else 'back'
            else:
                facing = 'front' if thumb_mcp[0] > pinky_mcp[0] else 'back'
    
    return direction, facing


def two_landmark_distance(vec1, vec2, dim=2):
    """ Calculate the distance between two landmarks. """
    vec = vec2[:dim] - vec1[:dim]
    distance = np.linalg.norm(vec)
    
    return distance


def calculate_angle(joints):
    """ Calculate the angle of three points. """
    vec1 = joints[0][:2] - joints[1][:2]
    vec2 = joints[2][:2] - joints[1][:2]

    cross = np.cross(vec1, vec2)
    dot = np.dot(vec1, vec2)
    angle = np.absolute(np.arctan2(cross, dot))

    return angle


def calculate_thumb_angle(joints, label, facing):
    """ Calculate the angle of three points. """
    vec1 = joints[0][:2] - joints[1][:2]
    vec2 = joints[2][:2] - joints[1][:2]

    if label == 'left':
        cross = np.cross(vec1, vec2) if facing == 'front' else np.cross(vec2, vec1)
    else:
        cross = np.cross(vec2, vec1) if facing == 'front' else np.cross(vec1, vec2)
    dot = np.dot(vec1, vec2)
    angle = np.arctan2(cross, dot)
    if angle < 0:
        angle += 2 * np.pi
    
    return angle


def get_finger_state(joint_angles, threshold):
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
    
    return finger_state


def map_gesture(gestures, finger_states, landmarks, wrist_angle, direction, boundary):
    """ Map detected gesture fetures to a pre-defined gesture template. """
    detected_gesture = None
    d = two_landmark_distance(landmarks[0], landmarks[5])
    thresh = d / 4
    for ges, temp in gestures.items():
        count = 0
        
        # check finger states
        flag = 0
        for i in range(len(finger_states)):
            if finger_states[i] not in temp['finger states'][i]:
                flag = 1
                break
        if flag == 0:
            count += 1
        # check wrist angle
        if temp['wrist angle'][0] < wrist_angle < temp['wrist angle'][1]:
            count += 1
        # check direction
        if temp['direction'] == direction:
            count += 1
        # check overlap
        if temp['overlap'] is None:
            count += 1
        else:
            flag = 0
            for lm1, lm2 in temp['overlap']:
                if two_landmark_distance(landmarks[lm1], landmarks[lm2]) > thresh:
                    flag = 1
                    break
            if flag == 0:
                count += 1
        # check boundary
        if temp['boundary'] is None:
            count += 1
        else:
            flag = 0
            for bound, lm in temp['boundary'].items():
                if boundary[bound] not in lm:
                    flag = 1
                    break
            if flag == 0:
                count += 1
        
        if count == 5:
            detected_gesture = ges
            break
    
    return detected_gesture


def draw_transparent_box(img, pt1, pt2, alpha=0.5, beta=0.5):
    """ Draw a transparent rectangle. """
    sub_img = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
    res = cv2.addWeighted(sub_img, alpha, white_rect, beta, 1.0)
    img[pt1[1]:pt2[1], pt1[0]:pt2[0]] = res


def draw_fingertips(landmarks, finger_states, img):
    """ Draw fingertips by finger states. """
    w = img.shape[1]
    r = int(w / 100)
    for i in range(5):
        fingertip = landmarks[4*(i+1)]
        color = THUMB_STATES[finger_states[i]][1] if i == 0 else NON_THUMB_STATES[finger_states[i]][1]
        cv2.circle(img, fingertip[:2], r, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, fingertip[:2], r+1, (255,255,255), int(r/5), lineType=cv2.LINE_AA)


def draw_bounding_box(landmarks, detected_gesture, img):
    """ Draw a bounding box of detected hand with gesture label. """
    w = img.shape[1]
    tor = int(w / 40)

    xs = landmarks[:,0]
    ys = landmarks[:,1]
    x_max, x_min = np.max(xs), np.min(xs)
    y_max, y_min = np.max(ys), np.min(ys)

    draw_transparent_box(img, (x_min-tor,y_min-tor-40), (x_max+tor,y_min-tor))

    cv2.rectangle(img, (x_min-tor,y_min-tor), (x_max+tor,y_max+tor),
                  LINE_COLOR, 1, lineType=cv2.LINE_AA)
    cv2.putText(img, f'{detected_gesture}', (x_min-tor+5,y_min-tor-10), 0, 1,
                LABEL_COLOR, 3, lineType=cv2.LINE_AA)


def display_hand_info(img, hand):
    """ Display hand information. """
    w = img.shape[1]
    tor = int(w /40)

    landmarks = hand['landmarks']
    label = hand['label']
    wrist_angle = hand['wrist_angle']
    direction = hand['direction']
    facing = hand['facing']

    xs = landmarks[:,0]
    ys = landmarks[:,1]
    x_max, x_min = np.max(xs), np.min(xs)
    y_max, y_min = np.max(ys), np.min(ys)

    cv2.rectangle(img, (x_min-tor,y_min-tor), (x_max+tor,y_max+tor),
                  LINE_COLOR, 1, lineType=cv2.LINE_AA)
    cv2.putText(img, f'LABEL: {label} hand', (x_min-tor,y_min-4*tor-10), 0, 0.6,
                LINE_COLOR, 2, lineType=cv2.LINE_AA)
    cv2.putText(img, f'DIRECTION: {direction}', (x_min-tor,y_min-3*tor-10), 0, 0.6,
                LINE_COLOR, 2, lineType=cv2.LINE_AA)
    cv2.putText(img, f'FACING: {facing}', (x_min-tor,y_min-2*tor-10), 0, 0.6,
                LINE_COLOR, 2, lineType=cv2.LINE_AA)
    cv2.putText(img, f'WRIST ANGLE: {round(wrist_angle,1)}', (x_min-tor,y_min-tor-10),
                0, 0.6, LINE_COLOR, 2, lineType=cv2.LINE_AA)


def draw_vol_bar(img, pt1, pt2, vol_bar, vol, fps, bar_x_range, activated):
    """ Draw a volume bar. """
    draw_transparent_box(img, pt1, pt2)
    
    cv2.putText(img, f'FPS: {int(fps)}', (50,50), 0, 0.8,
                LABEL_COLOR, 2, lineType=cv2.LINE_AA)
    if activated:
        cv2.putText(img, f'Activated!', (50,90), 0, 0.8,
                    BAR_COLOR, 2, lineType=cv2.LINE_AA)
    else:
        cv2.putText(img, f'Deactivated!', (50,90), 0, 0.8,
                    LABEL_COLOR, 2, lineType=cv2.LINE_AA)

    cv2.rectangle(img, (bar_x_range[0],110), (bar_x_range[1],130),
                                    BAR_COLOR, 1, lineType=cv2.LINE_AA)
    cv2.rectangle(img, (bar_x_range[0],110), (int(vol_bar),130),
                                    BAR_COLOR, -1, lineType=cv2.LINE_AA)
    cv2.putText(img, f'{int(vol)}', (bar_x_range[1]+20,128), 0, 0.8,
                                    LABEL_COLOR, 2, lineType=cv2.LINE_AA)


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
