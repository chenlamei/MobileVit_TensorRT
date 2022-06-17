import torch
import tqdm
import string
import numpy as np
import os.path as osp
import MobileViT_Pytorch.models as models
from imagenet_lmdb_datasets import get_test_data
valdir="/target/test_data/val/val.lmdb"
val_loader=get_test_data(valdir)
net = models.MobileViT_S()
state_dict = torch.load("MobileViT_Pytorch/weights-file/model_best.pth.tar", 
                        map_location=torch.device('cpu'))['state_dict']
for key in list(state_dict.keys()):
    state_dict[key.replace('module.', '')] = state_dict.pop(key)
net.load_state_dict(state_dict)
net.eval()
#gen test data for batchsize=1 2 4 8 
in_data=[];out_data=[];class_data=[]
for it, (inputs, targets) in enumerate(val_loader):
    out=net(inputs).detach().numpy()
    inputs = inputs.numpy()
    targets = targets.numpy()

    if it==0:#batchsize=1
        save_data={"in_tensor":inputs,"class":targets,"out_tensor":out}
        np.save("benchmark/data/batch1",save_data)
    elif it>0 and it<3:#batchsize=2
        in_data.append(inputs)
        class_data.append(targets)
        out_data.append(out)
        if it==2:
            save_data={"in_tensor":np.array(in_data).squeeze(),
                        "class":np.array(class_data).squeeze(),
                        "out_tensor":np.array(out_data).squeeze()}
            np.save("benchmark/data/batch2",save_data) 
            in_data=[];out_data=[];class_data=[]
    elif it>2 and it<7:#batchsize=4
        in_data.append(inputs)
        class_data.append(targets)
        out_data.append(out)
        if it==6:
            save_data={"in_tensor":np.array(in_data).squeeze(),
                        "class":np.array(class_data).squeeze(),
                        "out_tensor":np.array(out_data).squeeze()}
            np.save("benchmark/data/batch4",save_data) 
            in_data=[];out_data=[];class_data=[]
    elif it>6 and it<15:#batchsize=8
        in_data.append(inputs)
        class_data.append(targets)
        out_data.append(out)
        if it==14:
            save_data={"in_tensor":np.array(in_data).squeeze(),
                        "class":np.array(class_data).squeeze(),
                        "out_tensor":np.array(out_data).squeeze()}
            np.save("benchmark/data/batch8",save_data) 
            in_data=[];out_data=[];class_data=[]
    elif it>14 and it<115:#calibration data
        in_data.append(inputs)
        if it==114:
            save_data={"in_tensor":np.array(in_data).squeeze()}
            np.save("benchmark/data/calibration",save_data) 
            in_data=[]
    else:break



