import torch
import onnx
from onnxsim import simplify
import json
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import MobileViT_Pytorch.models as models

def convert_onnx_dynamic(model,save_path,simp=False):
    x = torch.randn(1, 3, 256,256)
    input_name = 'input'
    output_name = 'class'
    torch.onnx.export(model,x,save_path,input_names = [input_name],
                    output_names = [output_name],dynamic_axes= {
                        input_name: {0: 'B'},
                        output_name: {0: 'B'}}
                   )
    if simp:
      onnx_model = onnx.load(save_path) 
      model_simp, check = simplify(onnx_model,input_shapes={'input':(4,3,256,256)},dynamic_input_shape=True)
      assert check, "Simplified ONNX model could not be validated"
      onnx.save(model_simp, save_path)
      print('simplify onnx done')

def convert_onnx(model,save_path,batch=1,simp=False):
    
    input_names = ['input']
    output_names=["class"]
    x = torch.randn(batch, 3, 256,256)
    torch.onnx.export(model, x, save_path,input_names =input_names,output_names=output_names)
    if simp:
      onnx_model = onnx.load(save_path) 
      model_simp, check = simplify(onnx_model)
      assert check, "Simplified ONNX model could not be validated"
      onnx.save(model_simp, save_path)
      print('simplify onnx done')


def load_mobilevit_weights(model_path):
  # Create an instance of the MobileViT model
  net = models.MobileViT_S()
  state_dict = torch.load(model_path, map_location=torch.device('cpu'))['state_dict']
  for key in list(state_dict.keys()):
    state_dict[key.replace('module.', '')] = state_dict.pop(key)

  net.load_state_dict(state_dict)
  
  return net
if __name__=="__main__":
    net = load_mobilevit_weights("MobileViT_Pytorch/weights-file/model_best.pth.tar")
    convert_onnx(net,"./target/MobileViT.onnx")
    convert_onnx_dynamic(net,"./target/MobileViT_dynamic.onnx")