import cv2
import math


text_color = (255,0,0)       # text color
bar_color = (255,0,0)        # bar color
line_color = (255,255,255)   # landmark connection color
lm_color = (255,0,0)         # landmark color


def draw_vol_bar(img, vol_bar, vol, bar_x_range):
    cv2.rectangle(img, (bar_x_range[0],20), (bar_x_range[1],40), bar_color, 1, lineType=cv2.LINE_AA)
    cv2.rectangle(img, (bar_x_range[0],20), (int(vol_bar),40), bar_color, -1, lineType=cv2.LINE_AA)
    cv2.putText(img, f'{int(vol)}', (bar_x_range[1]+10,40), 0, 0.8, text_color, 2, lineType=cv2.LINE_AA)


def draw_landmarks(img, pt1, pt2, color=lm_color):
    cv2.circle(img, pt1, 10, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, pt2, 10, color, -1, lineType=cv2.LINE_AA)
    cv2.line(img, pt1, pt2, line_color, 3)


def two_landmark_distance(landmarks, id1, id2):
    x1, y1 = landmarks[id1][0], landmarks[id1][1]
    x2, y2 = landmarks[id2][0], landmarks[id2][1]
    length = math.hypot(x2 - x1, y2 - y1)
    
    return length, (x1,y1), (x2,y2)


def update_buffer(length, buffer_list, buffer):
    if len(buffer_list) < buffer:
        buffer_list.append(length)
    else:
        buffer_list.pop(0)
        buffer_list.append(length)
    
    return buffer_list


def check_buffer(buffer_list, direction):
    if direction == 1:
        return all(i < j for i, j in zip(buffer_list, buffer_list[1:]))
    if direction == -1:
        return all(i > j for i, j in zip(buffer_list, buffer_list[1:]))
