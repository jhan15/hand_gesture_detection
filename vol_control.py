import cv2
import numpy as np
import time
import math
from hand import handDetector
from osascript import osascript


def main():
    wCam, hCam = 640, 480
    cap = cv2.VideoCapture(1)
    cap.set(3, wCam)
    cap.set(4, hCam)

    pTime = 0
    cTime = 0

    detector = handDetector(detectCon=0.7)

    lm1, lm2 = 4, 8
    min_vol = 0
    max_vol = 100
    vol = 0
    volBar = 200

    textColor = (51,255,51)
    lmColor = (255,0,255)
    lineColor = (255,255,255)

    while True:
        _, img = cap.read()
        img = detector.findHands(img, draw=False)
        lmList = detector.findPosition(img, draw=False)
        
        if len(lmList) > 0:
            x1, y1 = lmList[lm1][1], lmList[lm1][2]
            x2, y2 = lmList[lm2][1], lmList[lm2][2]

            cv2.circle(img, (x1,y1), 10, lmColor, cv2.FILLED)
            cv2.circle(img, (x2,y2), 10, lmColor, cv2.FILLED)
            cv2.line(img, (x1,y1), (x2,y2), lineColor, 3)

            length = math.hypot(x2 - x1, y2 - y1)
            min_len = 20
            max_len = 150

            vol = np.interp(length, [min_len,max_len], [min_vol,max_vol])
            volBar = np.interp(length, [min_len,max_len], [200,450])

            osascript("set volume output volume {}".format(vol))
        
        cv2.rectangle(img, (200,20), (450,40), textColor, 2)
        cv2.rectangle(img, (200,20), (int(volBar),40), textColor, cv2.FILLED)
        cv2.putText(img, f'{int(vol)}', (460,40), cv2.FONT_HERSHEY_PLAIN, 2, textColor, 3)
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (30,40), cv2.FONT_HERSHEY_PLAIN, 2, textColor, 3)

        cv2.imshow('Img', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
    