import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2,
                min_detection_confidence=0.7, min_tracking_confidence=0.5):
        
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
                                    self.static_image_mode,
                                    self.max_num_hands,
                                    self.min_detection_confidence,
                                    self.min_tracking_confidence)
    
    def detect_hands(self, img):
        hands = None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        if self.results.multi_hand_landmarks:
            h, w, _ = img.shape
            num_hands = len(self.results.multi_hand_landmarks)
            hands = [None] * num_hands

            for i in range(num_hands):
                hands[i] = dict()
                handedness = self.results.multi_handedness[i]
                hand_landmarks = self.results.multi_hand_landmarks[i]

                hands[i]['index'] = handedness.classification[0].index
                hands[i]['label'] = handedness.classification[0].label

                lm_list = list()
                wrist_z = hand_landmarks.landmark[0].z

                for lm in hand_landmarks.landmark:
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)
                    cz = lm.z - wrist_z
                    lm_list.append([cx, cy, cz])
                
                hands[i]['lm'] = lm_list
        
        return img, hands
    
    def draw_landmarks(self, img):
        if self.results.multi_hand_landmarks:
            for landmarks in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(img, landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return img


def main():
    ptime = 0
    ctime = 0
    cap = cv2.VideoCapture(1)
    cap.set(3, 640)
    cap.set(4, 480)
    detector = HandDetector(min_detection_confidence=0.7)

    while True:
        _, img = cap.read()
        img = cv2.flip(img, 1)
        img, _ = detector.detect_hands(img)
        img = detector.draw_landmarks(img)
        
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
