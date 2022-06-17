import json
import onnx
import torch
import argparse
import torch.nn as nn
from onnxsim import simplify
from collections import OrderedDict
import torch.nn.functional as F
import MobileViT_Pytorch.models as models

def convert_onnx_dynamic(model,save_path,simp=False):
    x = torch.randn(4, 3, 256,256)
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

def get_net(model_path,opt_=False):
  if opt_:
    net = models.MobileViT_S_opt()
  else:
    net = models.MobileViT_S()
  state_dict = torch.load(model_path, 
                          map_location=torch.device('cpu'))['state_dict']
  model_state_dict=net.state_dict()
  for key in list(state_dict.keys()):
    if key[7:] in model_state_dict.keys():
      model_state_dict[key[7:]]=state_dict[key]
    
  net.load_state_dict(model_state_dict)
  net.eval()
  return net
if __name__=="__main__":
  parser = argparse.ArgumentParser(description='torch to onnx describe.')
  parser.add_argument(
      "--model_path",
      type = str,
      default="MobileViT_Pytorch/weights-file/model_best.pth.tar",
      help="torch weight path, default is MobileViT_Pytorch/weights-file/model_best.pth.tar.")

  parser.add_argument(
      "--save_path",
      type=str,
      default="./target/MobileViT.onnx",
      help="save direction of onnx models,default is ./target/MobileViT.onnx.")

  parser.add_argument(
      "--batch",
      type=int,
      default=1,
      help="batchsize of onnx models, default is 1.")

  parser.add_argument(
      "--opt",
      default=False, action='store_true',
      help="model optmization , default is False.")
  parser.add_argument(
      "--dynamic",
      default=False, action='store_true',
      help="export  dynamic onnx model , default is False.")

  args = parser.parse_args()
  print(args)
  net = get_net(args.model_path,opt_=args.opt)
  if args.dynamic:
    convert_onnx_dynamic(net,args.save_path,simp=True)
  else:
    convert_onnx(net,args.save_path,simp=True,batch=args.batch)
  
