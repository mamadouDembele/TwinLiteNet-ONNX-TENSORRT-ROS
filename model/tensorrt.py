import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

class TwinLiteNet():
    def __init__(self, model_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(model_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.inputs = []
        self.outputs = []
        self.allocations = []
        
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

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
        cuda.memcpy_htod(self.inputs[0]['allocation'], img)
        self.context.execute_v2(self.allocations)
        outputs = []
        for out in self.outputs:
            output = np.zeros(out['shape'],out['dtype'])
            cuda.memcpy_dtoh(output, out['allocation'])
            outputs.append(output)
        
        da = outputs[0]
        lanes = outputs[1]

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
