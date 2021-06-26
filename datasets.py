import glob
import random
import os
import time
import torch
import numpy
import numpy as np
numpy.set_printoptions(threshold=np.inf)
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import cv2
# class ImageDataset(Dataset):
#     def __init__(self,root, transforms_= None, unaligned = False):
#         self.data = pd.read_csv('fashion.csv')
#         self.data2 = pd.read_csv('atr.csv')
#         self.transform = transforms.Compose(transforms_)
#         self.unaligned = unaligned
#         self.len = len(self.data)
#
#     def __getitem__(self, index):
#         item_A = self.transform(Image.open(self.data["IMG_NAME"][index+965]))
#         item_B = self.transform(Image.open(self.data2["IMG_NAME"][index+965]).convert("RGB"))
#         return {'A': item_A, 'B': item_B}
#
#     def __len__(self):
#         return 500

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))
        self.files_C = sorted(glob.glob(os.path.join(root, '%s/C' % mode) + '/*.*'))

    def __getitem__(self, index):
        arr = np.array(Image.open(self.files_A[index % len(self.files_A)]).convert("RGB"))
        item_A = self.transform(Image.fromarray(arr))
        # if self.unaligned:
        #     item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        # else:
        item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert("RGB"))

        trans = transforms.Compose([transforms.ToTensor(), transforms.Resize(int(256), Image.BICUBIC),
                transforms.CenterCrop(256), transforms.Normalize((0.5), (0.5))])

        item_C = trans(Image.fromarray(np.array(Image.open(self.files_C[index % len(self.files_C)]))//255))
        # mat = Image.fromarray(np.array(Image.open(self.files_C[index % len(self.files_C)]))//255).convert('RGB')
        # item_C = self.transform(Image.fromarray(np.array(mat)*np.array(Image.open(self.files_A[index % len(self.files_A)]))))
        item_A = torch.cat((item_A,item_C),dim=0)
        item_B = torch.cat((item_B,item_C),dim=0)
        return {'A': item_A, 'B': item_B, 'C': item_C}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))