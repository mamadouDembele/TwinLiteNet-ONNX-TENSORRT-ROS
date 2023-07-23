#!/usr/bin/env python3
import rospy
import pycuda.driver as cuda
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from model.tensorrt import TwinLiteNet



class LaneDetection:
    def __init__(self):

        model_engine = "/home/mamadou/TwinLiteNet-ONNX-TENSORRT-ROS/pretrained/best.engine"
        
        #model_engine = rospy.get_param("model_path")
        rospy.loginfo("model_engine_path: %s", model_engine)
        
        self.cuda_ctx = cuda.Device(0).make_context()
        self.image_pub = rospy.Publisher("~image", Image, queue_size=1)
        self.bridge = CvBridge()
        self.model = TwinLiteNet(model_engine)
        self.image_sub = rospy.Subscriber("/camera_front_center_wide/image_raw", Image, self.callback)
    

    def callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return
        
        self.cuda_ctx.push()
        output_image = self.model.forward(cv_image)

        if self.image_pub.get_num_connections() > 0:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(output_image, "bgr8"))
        
        self.cuda_ctx.pop()

    
if __name__ == '__main__':
    rospy.init_node('lane_recognition_node')
    LaneDetection()
    rospy.spin()
