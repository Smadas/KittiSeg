#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image


def callback(data):
    rospy.loginfo(rospy.get_caller_id())


def path_seg():
    rospy.init_node('path_seg', anonymous=True)

    rospy.Subscriber("/usb_cam/image_raw", Image, callback)

    rospy.spin()


if __name__ == '__main__':
    path_seg()