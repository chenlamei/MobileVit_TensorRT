import os
import torch
import ctypes
import numpy as np
from glob import glob 
from cuda import cudart
import tensorrt as trt
import MobileViT_Pytorch.utils as utils
import MobileViT_Pytorch.models as models
from imagenet_lmdb_datasets import get_test_data
from convert_to_onnx import get_net


if __name__=='__main__':
    batch=8
    val_acc1 = utils.AverageMeter("Val Acc@1", ":6.2f")
    val_acc5 = utils.AverageMeter("Val Acc@5", ":6.2f")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    valdir="/target/test_data/val/val.lmdb"
    val_loader=get_test_data(valdir,batch)
    model_path="MobileViT_Pytorch/weights-file/model_best.pth.tar"
    net=get_net(model_path).to(device)
    for it, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        output=net(inputs)
        acc1, acc5 = utils.accuracy(output, targets, topk=(1, 5))
        val_acc1.update(acc1.item(), batch)
        val_acc5.update(acc5.item(), batch)
    print("top1:", val_acc1.avg,"top5:",val_acc5.avg)
    