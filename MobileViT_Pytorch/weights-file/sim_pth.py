import torch
model_path='./model_best.pth.tar'
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
state_dict.pop('optimizer')
torch.save(state_dict,"model_best.pth.tar")