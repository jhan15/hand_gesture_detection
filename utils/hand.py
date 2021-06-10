import cv2
import mediapipe as mp
import time


class handDetector:
    def __init__(self, mode=False, max_hands=2, detect_con=0.7, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detect_con = detect_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands,
                                        self.detect_con, self.track_con)
        self.mp_drawing = mp.solutions.drawing_utils
        self.results = None
    
    def find_hands(self, img, draw=True):
        num_hands = 0
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks:
            num_hands = len(self.results.multi_hand_landmarks)
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return img, num_hands
    
    def find_position(self, img, hand_id=0, draw=True):
        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_id]
            for id, lm in enumerate(my_hand.landmark):
                h, w, _ = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z
                lm_list.append([id, cx, cy, cz])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255,0,255), cv2.FILLED)

        return lm_list


def main():
    ptime = 0
    ctime = 0
    cap = cv2.VideoCapture(1)
    detector = handDetector(detect_con=0.7)

    while True:
        _, img = cap.read()
        img = cv2.flip(img, 1)
        img, num_hands = detector.find_hands(img)
        if num_hands > 0:
            for i in range(num_hands):
                _ = detector.find_position(img, i)
        
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, f'FPS: {int(fps)}', (30,40), cv2.FONT_HERSHEY_PLAIN, 3, (51,255,51), 3)

        cv2.imshow('Img', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
