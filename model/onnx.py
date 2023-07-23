import os
import sys
import cv2
import numpy as np
import onnxruntime
import time

class TwinLiteNet():
    def __init__(self, model_path):
        self.ort_session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_width = 640
        self.input_height = 360
   
    def forward(self, img):
        start_time = time.time()
        img_ = img.copy()        
        img = cv2.resize(img, (self.input_width, self.input_height))
        img_rs=img.copy()

        img = img.astype(np.float32) / 255.0 

        img = np.transpose(np.float32(img[:,:,:,np.newaxis]), (3,2,0,1))
        img = np.ascontiguousarray(img)

        ort_inputs = {self.ort_session.get_inputs()[0].name: img}
        ort_outs = self.ort_session.run(None, ort_inputs)
        
        da = ort_outs[0]
        lanes = ort_outs[1]

        da = np.argmax(da, 1)
        lanes = np.argmax(lanes, 1)
        
        da = da.astype('uint8')
        da = da[0]*255

        lanes = lanes.astype('uint8')
        lanes = lanes[0]*255

        img_rs[da>100]=[255,0,0]
        img_rs[lanes>100]=[0,255,0]
        elapsed_time = time.time() - start_time

        cv2.putText(
            img_rs,
            "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return img_rs