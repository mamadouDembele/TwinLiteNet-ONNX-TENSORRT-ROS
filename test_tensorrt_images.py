import cv2
import os
from model.tensorrt import TwinLiteNet

if __name__ == "__main__":
    model_path = 'pretrained/best.engine'
    model = TwinLiteNet(model_path)
    image_list=os.listdir('images')
    for i, imgName in enumerate(image_list):
        img = cv2.imread(os.path.join('images',imgName))
        output = model.forward(img)
        cv2.imwrite(os.path.join('results',imgName),output)
    
    print("Done !!!")