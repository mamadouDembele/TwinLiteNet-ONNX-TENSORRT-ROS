import cv2
import os
from model.onnx import TwinLiteNet

if __name__ == "__main__":
    model_path = 'pretrained/best.onnx'
    model = TwinLiteNet(model_path)
    video_path = 'video/sample.mp4'
    video_capture = cv2.VideoCapture(video_path)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        output_image = model.forward(frame)
        
        cv2.imshow("TwinLiteNet", output_image)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
 
    video_capture.release()
    cv2.destroyAllWindows()