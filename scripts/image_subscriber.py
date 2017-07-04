#!/usr/bin/env python

from __future__ import print_function

import roslib
roslib.load_manifest('benchmarking_simulator')
import sys
import rospy
import cv2
import numpy as np

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from sensor_msgs.msg import Image

import image_net_trainer as net


class image_subscriber:

  def __init__(self):
    self.image_pub = rospy.Publisher("image_topic_2",Image,queue_size=10)

    self.bridge = CvBridge()
    
    self.depth_image_sub = message_filters.Subscriber('depth_image', Image)
    self.rgb_image_sub = message_filters.Subscriber('rgb_image', Image)

    self.ts = message_filters.ApproximateTimeSynchronizer([self.depth_image_sub,self.rgb_image_sub],10,0.1)
    self.ts.registerCallback(self.callback)

  def callback(self,depth_data,rgb_data):
    try:
      cv_depth_image = self.bridge.imgmsg_to_cv2(depth_data,desired_encoding="8UC1")
      cv_rgb_image = self.bridge.imgmsg_to_cv2(rgb_data,desired_encoding="bgr8")
    except CvBridgeError as e:
      print(e)

    result=self.stub(cv_rgb_image)

    self.filter(cv_depth_image,result)
    
  def stub(self,image):
    #cv2.imshow("Image window",image)
    #cv2.waitKey(30)
    #rospy.loginfo("image size: %d",image.size)

    output=[0,1,0,50,100,50,100]
    return output


  def filter(self,image,result):
    roi=image[result[3]:result[4],result[5]:result[6]]
    #cv2.imshow("Image window",roi)
    #cv2.waitKey(30)
    #rospy.loginfo("roi image size: %s",str(roi.shape))
    
    max_depth=np.amax(roi)
    #rospy.loginfo("max depth: %d",max_depth)
    
    ret,output=cv2.threshold(image,max_depth,255,cv2.THRESH_TOZERO_INV)
    #cv2.imshow("Image window",output)
    #cv2.waitKey(30)
    #rospy.loginfo("output image size: %s",str(output.shape))
    
    return output

def main(args):
  rospy.init_node('image_subscriber', anonymous=True)
  im_sub = image_subscriber()
  rospy.loginfo("Starting image_subcriber node")
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
