#!/usr/bin/env python

from __future__ import print_function

import roslib
roslib.load_manifest('benchmarking_simulator')
import sys
import rospy
import cv2

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from sensor_msgs.msg import Image

class image_subscriber:

  def __init__(self):
    self.image_pub = rospy.Publisher("image_topic_2",Image,queue_size=10)

    self.bridge = CvBridge()
    
    self.depth_image_sub = message_filters.Subscriber('depth_image', Image)
    self.rgb_image_sub = message_filters.Subscriber('rgb_image', Image)

    self.ts = message_filters.ApproximateTimeSynchronizer([self.depth_image_sub,self.rgb_image_sub],10,0.3)
    self.ts.registerCallback(self.callback)

  def callback(self,depth_data,rgb_data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")
    except CvBridgeError as e:
      print(e)


    cv2.imshow("Image window", cv_image)
    cv2.waitKey(3)

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)

def main(args):
  im_sub = image_subscriber()
  rospy.init_node('image_subscriber', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
