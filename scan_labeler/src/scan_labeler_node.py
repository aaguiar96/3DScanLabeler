#!/usr/bin/env python

import rospy
import sys
from sensor_msgs.msg import PointCloud2

class ScanLabeler:
    # Class constructor
    # - get topic names and bringup callback functions
    def __init__(self):
        self.sub   = rospy.Subscriber("velodyne_points", PointCloud2, self.scanCallback)

    # Callback function to read 3D scan measures
    def scanCallback(self, data):
        print("Receiving data...")

def main(args):
    # Instantiate class and bringup ROS node
    scan_handler = ScanLabeler()
    rospy.init_node("scan_labeler")

    # ROS spin until user break it
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)
