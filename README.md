[![GitHub issues](https://img.shields.io/github/issues/jhan15/hand_gesture_detection)](https://github.com/jhan15/hand_gesture_detection/issues)
![GitHub last commit](https://img.shields.io/github/last-commit/jhan15/hand_gesture_detection?color=ff69b4)

# hand_gesture_detection
It's a hand gesture detection project based on the hand module of Google's [mediapipe](https://github.com/google/mediapipe) API. The hand module gives the coordinates of 21 hand landmarks, which can be found in the image below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/62132206/124274282-5af07f80-db41-11eb-9ac8-bf14d9680d68.png?raw=true" width="600">
</p>

Functionalities included in the project are:
1. Hand detection;
2. Hand gesture detection;
3. Volume control using hand gestures.

## Requirements
Python 3.8 or later with dependencies listed in [requirements.txt](https://github.com/jhan15/gesture_detection/blob/master/requirements.txt). To install run:

```bash
$ git clone https://github.com/jhan15/hand_gesture_detection.git
$ cd hand_gesture_detection
$ pip install -r requirements.txt
```

## Usage

### Hand detector
Detect hands on streams, it draws the landmarks on detected hands and returns hand features, including handedness, landmark coordinates, direction, facing, boundary, wrist angle.

```bash
$ python3 hand.py --max_hands 2
```

### Gesture detector
Detect hand gestures on streams, it can detect a series of pre-defined hand gestures, including Chinese number 1 - 10, 'OK', 'Thumbs-up', 'Thumbs-down', 'Rock', 'Claw', 'C' shape, 'O' shape, 'pinch'.

```bash
$ python3 gesture.py --mode single
```

(Currenly only single-hand gestures are supported, double-hand gestures TBD)

### Volume controller
Control volume using hand gestures. It's a possible application in areas like smart home and in-cabin interaction. Hand gestures can be used for activation, deactivation, and control.

```bash
$ python3 vol_controller.py --control continuous # continuous control
                                      step # step control
```

## Demo

### Hand detector

### Gesture detector

### Volume controller

#### Continuous control

![vol1](https://user-images.githubusercontent.com/62132206/121547644-9a2d2400-ca0c-11eb-9141-a280243f71b0.gif)

#### Step control

![vol3](https://user-images.githubusercontent.com/62132206/121547653-9c8f7e00-ca0c-11eb-9319-e75a4e96cf6f.gif)

#### Issue

The package I used to control Macbook's volume is [osascript](https://github.com/andrewp-as-is/osascript.py), it will reduce the FPS from ~30 to ~5.
