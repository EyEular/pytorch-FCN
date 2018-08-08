import os
from PIL import Image
from torch.utils import data
import numpy as np
import pandas as pd
from torchvision import transforms as T

import sys
sys.path.append("..")
from utils import *



class KaggleSalt(data.Dataset):
    def __init__(self, root, train = True):
        self.root = root
        self.train_root = root + '/train/images'
        self.imgs = [img for img in os.listdir(self.train_root)]
        print(len(self.imgs))
        
        self.df = pd.read_csv(self.root + '/train.csv')
        print(self.df.columns)

        # define the transform
        self.transforms = T.Compose([
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        #self.__getitem__(1)
        #import ipdb
        #ipdb.set_trace()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_name = self.imgs[index]
        # img_name = '9842f69f8d.png'
        # print(img_name)
        label_df = self.df.loc[self.df['id'] == img_name.split('.')[0]]
        
        data = Image.open(self.root + '/train/images/' + img_name)
        data = self.transforms(data)

        #import ipdb
        #ipdb.set_trace()

        label = rle_decode(data.size(), str(label_df.iloc[0,1]))
        
        #print(mask_compare(label, self.root + '/train/masks/' + img_name))
        #import ipdb
        #ipdb.set_trace()
        return data, label


        

