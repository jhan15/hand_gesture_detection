[![GitHub issues](https://img.shields.io/github/issues/jhan15/hand_gesture_detection)](https://github.com/jhan15/hand_gesture_detection/issues)
![GitHub last commit](https://img.shields.io/github/last-commit/jhan15/hand_gesture_detection?color=ff69b4)

# hand_gesture_detection
Hand gesture detection based on the hand module of Google's [mediapipe](https://github.com/google/mediapipe) API. The hand module gives the coordinates of 21 hand landmarks, which can be found in the image below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/62132206/124274282-5af07f80-db41-11eb-9ac8-bf14d9680d68.png?raw=true" width="600">
</p>

This project focuses on three functionalities:
1. Hand detection.
2. Hand gesture detection.
3. Volume control using hand gestures.

## Requirements
Python 3.8 or later with dependencies listed in [requirements.txt](https://github.com/jhan15/gesture_detection/blob/master/requirements.txt). To install run:

```bash
$ git clone https://github.com/jhan15/hand_gesture_detection.git
$ cd hand_gesture_detection
$ pip install -r requirements.txt
```

## Usage

```bash
# Hand detector
$ python3 hand.py --max_hands 2

# Gesture detector
$ python3 gesture.py --mode single # currenly only single-hand gestures are supported

# Volume controller
$ python3 vol_controller.py --control continuous # continuous control
                                      step # step control
```

## Demo

### Hand detector
Detect hands on streams, it draws the landmarks on detected hands and returns several hand features, including handedness, landmark coordinates, hand direction, hand facing, boundary landmarks, wrist angle.

![hand1](https://user-images.githubusercontent.com/62132206/127870204-96725670-6db0-4025-be46-bd3efacae085.gif)

### Gesture detector
Detect hand gestures on streams, now it can detect 18 pre-defined hand gestures, including Chinese number 1 - 10, 'OK', 'Thumbs-up', 'Thumbs-down', 'Rock', 'Claw', 'C shape', 'O shape', 'Pinch'.

![gesture1](https://user-images.githubusercontent.com/62132206/127870254-c205a04a-4b7f-4ce3-b4e6-549a38183125.gif)

### Volume controller
Control volume using hand gestures. It's a possible application in areas like smart home and in-cabin interaction. Hand gestures can be used for activation, control, and deactivation. In this case, we use 'Pinch' as activation and control gesture and 'C shape' as deactivation gesture. There are two types of control behavior here, continuous control and step control.

#### Continuous control

![continuous1](https://user-images.githubusercontent.com/62132206/127870281-51b98ccb-60c0-491d-b26e-712804d0b639.gif)

#### Step control

![step1](https://user-images.githubusercontent.com/62132206/127870295-00b94af1-3fbb-474b-b0f0-3e717d6d4882.gif)

#### Issue

The package I used to control Macbook's volume is [osascript](https://github.com/andrewp-as-is/osascript.py), it will reduce the FPS from ~30 to ~5.
