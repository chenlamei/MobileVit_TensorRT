
import os
import numpy as np
from cuda import cudart
import tensorrt as trt


ioFile="benchmark/data/calibration.npy"
ioData = np.load(ioFile,allow_pickle=True).item()
in_tensor=ioData['in_tensor']

class MobileVitCalibrator(trt.IInt8EntropyCalibrator2):

    def __init__(self, calibrationCount=25, inputShape=[4,3,256,256],cacheFile='./target/mobilevit.cacheFile'):
        trt.IInt8EntropyCalibrator2.__init__(self)
        
        self.shape = inputShape
        self.cacheFile = cacheFile
        self.calibrationCount = calibrationCount
        self.buffeSize = trt.volume(inputShape) * trt.float32.itemsize
        self.dIn =[] 
        self.dIn.append(cudart.cudaMalloc(self.buffeSize)[1])
        self.count = 0

    def __del__(self):
        cudart.cudaFree(self.dIn[0])
    def get_batch_size(self):  # do NOT change name
        return self.shape[0]

    def get_batch(self, nameList=None):  # do NOT change name
        if self.count < self.calibrationCount:
            start_idx = self.count*self.shape[0]
            end_idx = start_idx + self.shape[0]
            in_data=in_tensor[start_idx:end_idx,...]
            cudart.cudaMemcpy(self.dIn[0], in_data.ctypes.data, self.buffeSize, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            self.count += 1
            return self.dIn
        else:
            return None

    def read_calibration_cache(self):  # do NOT change name
        if os.path.exists(self.cacheFile):
            print("Succeed finding cahce file: %s" % (self.cacheFile))
            with open(self.cacheFile, "rb") as f:
                cache = f.read()
                return cache
        else:
            print("Failed finding int8 cache!")
            return

    def write_calibration_cache(self, cache):  # do NOT change name
        with open(self.cacheFile, "wb") as f:
            f.write(cache)
        print("Succeed saving int8 cache!")



if __name__ == "__main__":
    cudart.cudaDeviceSynchronize()
    m = MobileVitCalibrator()
    m.get_batch("ttt")
    
