# 3DScanLabeler

## Dependencies

`sudo apt install ros-<version>-ros-numpy`

## How to run

```
roslaunch scan_labeler playback.launch bag:=bag_file_name
```

* When rviz open, use `Publish Point` to click on a point close to the middle of the
  chessboard.
* Then, run the labeler

```
rosrun scan_labeler scan_labeler_node.py
```
