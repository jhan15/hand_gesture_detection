import cv2
import numpy as np
import time
import math
from utils import hand
from osascript import osascript


class volControl:
    def __init__(self, text_color=(51,255,51), lm_color=(255,0,255),
                line_color=(255,255,255), bar_color=(51,255,51)):

        self.w_cam = 640
        self.h_cam = 480
        self.cap = cv2.VideoCapture(1)
        self.cap.set(3, self.w_cam)
        self.cap.set(4, self.h_cam)
        self.detector = hand.handDetector()

        self.text_color = text_color
        self.lm_color = lm_color
        self.line_color = line_color
        self.bar_color = bar_color

        self.vol_range = [0, 100]
        self.bar_x = [200, 450]
        self.bar_y = [20, 40]
        self.vol = (self.vol_range[0] + self.vol_range[1]) // 2
        self.volBar = (self.bar_x[0] + self.bar_x[1]) // 2

    def pinch_length_control(self):
        lms = [4, 8]
        len_range = [20, 150] # length range, need to test to get the values
        ptime = 0
        ctime = 0

        while True:
            _, img = self.cap.read()
            img = cv2.flip(img, 1)
            img, _ = self.detector.find_hands(img, draw=False)
            lm_list = self.detector.find_position(img, draw=False)
            
            if len(lm_list) > 0:
                x1, y1 = lm_list[lms[0]][1], lm_list[lms[0]][2]
                x2, y2 = lm_list[lms[1]][1], lm_list[lms[1]][2]

                cv2.circle(img, (x1,y1), 10, self.lm_color, cv2.FILLED)
                cv2.circle(img, (x2,y2), 10, self.lm_color, cv2.FILLED)
                cv2.line(img, (x1,y1), (x2,y2), self.line_color, 3)

                length = math.hypot(x2 - x1, y2 - y1)

                self.vol = np.interp(length, len_range, self.vol_range)
                self.volBar = np.interp(length, len_range, self.bar_x)

                osascript("set volume output volume {}".format(self.vol))
            
            cv2.rectangle(img, (self.bar_x[0],self.bar_y[0]),
                            (self.bar_x[1],self.bar_y[1]), self.text_color, 2)
            cv2.rectangle(img, (self.bar_x[0],self.bar_y[0]),
                            (int(self.volBar),self.bar_y[1]), self.text_color, cv2.FILLED)
            cv2.putText(img, f'{int(self.vol)}', (self.bar_x[1]+10,self.bar_y[1]),
                            cv2.FONT_HERSHEY_PLAIN, 2, self.text_color, 3)
            
            ctime = time.time()
            fps = 1 / (ctime - ptime)
            ptime = ctime

            cv2.putText(img, f'FPS: {int(fps)}', (30,40), cv2.FONT_HERSHEY_PLAIN,
                        2, self.text_color, 3)

            cv2.imshow('Pinch length control', img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
    
    def pinch_open_close_control(self, vol_step=10, buffer=30):
        lms=[4, 8]
        up_range = [70, 100] # length range, need to test to get the values
        down_range = [20, 50] # length range, need to test to get the values

        ptime = 0
        ctime = 0

        buffer_list = []

        while True:
            _, img = self.cap.read()
            img = cv2.flip(img, 1)
            img, _ = self.detector.find_hands(img, draw=False)
            lm_list = self.detector.find_position(img, draw=False)

            if len(lm_list) > 0:
                x1, y1 = lm_list[lms[0]][1], lm_list[lms[0]][2]
                x2, y2 = lm_list[lms[1]][1], lm_list[lms[1]][2]

                cv2.circle(img, (x1,y1), 10, self.lm_color, cv2.FILLED)
                cv2.circle(img, (x2,y2), 10, self.lm_color, cv2.FILLED)
                cv2.line(img, (x1,y1), (x2,y2), self.line_color, 3)

                length = math.hypot(x2 - x1, y2 - y1)
                if len(buffer_list) < buffer:
                    buffer_list.append(length)
                else:
                    buffer_list.pop(0)
                    buffer_list.append(length)
                
                if len(buffer_list) == buffer and length > up_range[1]:
                    up = False
                    i = buffer - 2
                    while True:
                        if buffer_list[i] > buffer_list[i+1]:
                            break
                        if i == 0:
                            break
                        if buffer_list[i] < up_range[0]:
                            up = True
                            break
                        i -= 1
                    if up:
                        self.vol += vol_step
                        self.vol = min(self.vol, self.vol_range[1])
                        self.volBar = np.interp(self.vol, self.vol_range, self.bar_x)
                        buffer_list = []
                        
                        osascript("set volume output volume {}".format(self.vol))
                
                if len(buffer_list) == buffer and length < down_range[0]:
                    down = False
                    i = buffer - 2
                    while True:
                        if buffer_list[i] < buffer_list[i+1]:
                            break
                        if i == 0:
                            break
                        if buffer_list[i] > down_range[1]:
                            down = True
                            break
                        i -= 1
                    if down:
                        self.vol -= vol_step
                        self.vol = max(self.vol, self.vol_range[0])
                        self.volBar = np.interp(self.vol, self.vol_range, self.bar_x)
                        buffer_list = []
                        
                        osascript("set volume output volume {}".format(self.vol))
            
            cv2.rectangle(img, (self.bar_x[0],self.bar_y[0]),
                            (self.bar_x[1],self.bar_y[1]), self.text_color, 2)
            cv2.rectangle(img, (self.bar_x[0],self.bar_y[0]),
                            (int(self.volBar),self.bar_y[1]), self.text_color, cv2.FILLED)
            cv2.putText(img, f'{int(self.vol)}', (self.bar_x[1]+10,self.bar_y[1]),
                            cv2.FONT_HERSHEY_PLAIN, 2, self.text_color, 3)
            
            ctime = time.time()
            fps = 1 / (ctime - ptime)
            ptime = ctime

            cv2.putText(img, f'FPS: {int(fps)}', (30,40), cv2.FONT_HERSHEY_PLAIN, 2, self.text_color, 3)

            cv2.imshow('Pinch open close control', img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break

    # def four_finger_wave_control(self, lm_0=4, lms_1=[8, 12, 16, 20], lms_2=[7, 11, 15, 19],
    #                             lms_3=[6, 10, 14, 18], lms_4=[5, 9, 13, 17], vol_step=10, buffer=30):
    #     vol_range = [0, 100]
    #     bar_x = [200, 450]
    #     bar_y = [20, 40]
    #     vol = (vol_range[0] + vol_range[1]) // 2
    #     volBar = (bar_x[0] + bar_x[1]) // 2

    #     ptime = 0
    #     ctime = 0

    #     while True:
    #         _, img = self.cap.read()
    #         img = cv2.flip(img, 1)
    #         img, _ = self.detector.find_hands(img, draw=False)
    #         lm_list = self.detector.find_position(img, draw=False)

    #         if len(lm_list) > 0:
    #             x0, y0 = lm_list[lm_0][1], lm_list[lm_0][2]
    #             cv2.circle(img, (x0,y0), 10, self.lm_color, cv2.FILLED)
    #             for i in range(len(lms_1)):
    #                 x1, y1 = lm_list[lms_1[i]][1], lm_list[lms_1[i]][2]
    #                 x2, y2 = lm_list[lms_2[i]][1], lm_list[lms_2[i]][2]
    #                 x3, y3 = lm_list[lms_3[i]][1], lm_list[lms_3[i]][2]
    #                 x4, y4 = lm_list[lms_4[i]][1], lm_list[lms_4[i]][2]
    #                 cv2.circle(img, (x1,y1), 10, self.lm_color, cv2.FILLED)
    #                 cv2.circle(img, (x2,y2), 10, self.bar_color, cv2.FILLED)
    #                 cv2.circle(img, (x3,y3), 10, (51,51,255), cv2.FILLED)
    #                 cv2.circle(img, (x4,y4), 10, (255,51,51), cv2.FILLED)
    #                 cv2.line(img, (x1,y1), (x2,y2), self.line_color, 3)
    #                 cv2.line(img, (x2,y2), (x3,y3), self.line_color, 3)
    #                 cv2.line(img, (x3,y3), (x4,y4), self.line_color, 3)
                
            
    #         cv2.rectangle(img, (bar_x[0],bar_y[0]), (bar_x[1],bar_y[1]), self.text_color, 2)
    #         cv2.rectangle(img, (bar_x[0],bar_y[0]), (int(volBar),bar_y[1]), self.text_color, cv2.FILLED)
    #         cv2.putText(img, f'{int(vol)}', (bar_x[1]+10,bar_y[1]), cv2.FONT_HERSHEY_PLAIN,
    #                     2, self.text_color, 3)
            
    #         ctime = time.time()
    #         fps = 1 / (ctime - ptime)
    #         ptime = ctime

    #         cv2.putText(img, f'FPS: {int(fps)}', (30,40), cv2.FONT_HERSHEY_PLAIN, 2, self.text_color, 3)

    #         cv2.imshow('Four finger wave control', img)
    #         key = cv2.waitKey(1)
    #         if key == ord('q'):
    #             cv2.destroyAllWindows()
    #             break


if __name__ == '__main__':
    vc = volControl()
    vc.pinch_open_close_control()
    