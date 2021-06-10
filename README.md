# gesture_vol_control
It's a project to control the volume using different gestures. Gesture detection using Google's [mediapipe API](https://github.com/google/mediapipe).


## Requirements

Python 3.8 or later with dependencies listed in [requirements.txt](https://github.com/jhan15/gesture_detection/blob/master/requirements.txt). To install run:

```bash
$ pip install -r requirements.txt
```

## Usage

```bash
$ python3 vol_control.py
```

### Demo

You can watch the [video with audio](https://www.youtube.com/watch?v=l3ukvTslEB0).

![v1](https://user-images.githubusercontent.com/62132206/120515147-54e67200-c3ce-11eb-919d-4c732efb9c62.gif)

### Problem

The package I used to control Macbook's volume is [osascript](https://github.com/andrewp-as-is/osascript.py), it will reduce the FPS from ~30 to ~5.
