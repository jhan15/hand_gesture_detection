[![GitHub issues](https://img.shields.io/github/issues/jhan15/hand_gesture_detection)](https://github.com/jhan15/hand_gesture_detection/issues)
![GitHub last commit](https://img.shields.io/github/last-commit/jhan15/hand_gesture_detection?color=ff69b4)

# hand_gesture_detection
It's a project of exploring hand gesture detection and possible applications. It's based on Google's [mediapipe](https://github.com/google/mediapipe) API's hand module for hand detection, which provides coordinates of 21 hand landmarks.

<p align="center">
  <img src="https://user-images.githubusercontent.com/62132206/124274282-5af07f80-db41-11eb-9ac8-bf14d9680d68.png?raw=true" width="600">
</p>

### Gestures

<p align="center">
  <img src="https://user-images.githubusercontent.com/62132206/124275863-6349ba00-db43-11eb-84bb-bf7400b38173.png?raw=true" width="600">
</p>


## Requirements
Python 3.8 or later with dependencies listed in [requirements.txt](https://github.com/jhan15/gesture_detection/blob/master/requirements.txt). To install run:

```bash
$ git clone https://github.com/jhan15/hand_gesture_detection.git
$ cd hand_gesture_detection
$ pip install -r requirements.txt
```

## Usage

### Hand detector

```bash
$ python3 hand.py --max_hands 2
```

### Gesture detector

```bash
$ python3 gesture.py --mode single
```

## Application
### Volume controller

```bash
$ python3 vol_controller.py --control continuous # continuous control
                                      step # step control
```

### Demo

#### Continuous control by length

![vol1](https://user-images.githubusercontent.com/62132206/121547644-9a2d2400-ca0c-11eb-9141-a280243f71b0.gif)

#### Step control by open/close

![vol3](https://user-images.githubusercontent.com/62132206/121547653-9c8f7e00-ca0c-11eb-9319-e75a4e96cf6f.gif)

You can watch the [video with audio](https://www.youtube.com/watch?v=l3ukvTslEB0).

### Issue

The package I used to control Macbook's volume is [osascript](https://github.com/andrewp-as-is/osascript.py), it will reduce the FPS from ~30 to ~5.
