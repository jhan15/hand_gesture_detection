# gesture_vol_control
It's a project to control the volume using different gestures. Gesture detection using Google's [mediapipe API](https://github.com/google/mediapipe).


## Requirements

Python 3.8 or later with dependencies listed in [requirements.txt](https://github.com/jhan15/gesture_detection/blob/master/requirements.txt). To install run:

```bash
$ git clone https://github.com/jhan15/gesture_vol_control.git
$ cd gesture_vol_control
$ pip install -r requirements.txt
```

## Usage

```bash
$ python3 vol_control.py --control 'pinch_conti' # continuous control
                                   'pinch_step'  # step control
                                   'none'        # draw hand landmarks
```

### Demo

#### Continuous control by length

![vol1](https://user-images.githubusercontent.com/62132206/121547644-9a2d2400-ca0c-11eb-9141-a280243f71b0.gif)

#### Step control by open/close

![vol3](https://user-images.githubusercontent.com/62132206/121547653-9c8f7e00-ca0c-11eb-9319-e75a4e96cf6f.gif)

You can watch the [video with audio](https://www.youtube.com/watch?v=l3ukvTslEB0).

### Issue

The package I used to control Macbook's volume is [osascript](https://github.com/andrewp-as-is/osascript.py), it will reduce the FPS from ~30 to ~5.
