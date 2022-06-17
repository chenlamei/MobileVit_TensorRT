import six
import lmdb
import torch
import pickle
import msgpack
import tqdm
import string
import numpy as np
from PIL import Image
import pyarrow as pa
import os.path as osp
import torch.utils.data as data
import torchvision.datasets as datasets
from torchvision import transforms as T
import MobileViT_Pytorch.models as models
class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length =pa.deserialize(txn.get(b'__len__'))
            self.keys= pa.deserialize(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)

        # load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)   

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def get_test_data(valdir,batchsize=1):

    normalize = T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    val_dataset = ImageFolderLMDB(
        valdir,
        T.Compose([
            T.Resize((256,256)),
            T.ToTensor(),
            normalize,
    ]))
    val_loader = data.DataLoader(
        val_dataset, batch_size = batchsize, shuffle = True,
        num_workers = min(batchsize,4), pin_memory = False
    )
    return val_loader