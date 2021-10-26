import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root,count = None,transforms_1=None,transforms_2=None, unaligned=False):
        self.transform1 = transforms.Compose(transforms_1)
        self.transform2 = transforms.Compose(transforms_2)
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))
        self.unaligned = unaligned
        
    def __getitem__(self, index):
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        
        item_A = self.transform1(np.load(self.files_A[index % len(self.files_A)]).astype(np.float32))
        #random.seed(seed)
        if self.unaligned:
            item_B = self.transform2(np.load(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        
        else:  
            item_B = self.transform2(np.load(self.files_B[index % len(self.files_B)]).astype(np.float32))
        return {'A': item_A, 'B': item_B}
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class ValDataset(Dataset):
    def __init__(self, root,count = None,transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))
        
    def __getitem__(self, index):
        item_A = self.transform(np.load(self.files_A[index % len(self.files_A)]).astype(np.float32))
        if self.unaligned:
            item_B = self.transform(np.load(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(np.load(self.files_B[index % len(self.files_B)]).astype(np.float32))
        return {'A': item_A, 'B': item_B}
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))