import os
import torch
import zipfile
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
from datasets import load_dataset

    
class LaionV2:
    def __init__(self, metadata_path, folder_path, fid='folder', key='key', caption_col=['caption', 'prompt'], p=[0.2, 0.8], transform=None):
        self.df = pd.read_parquet(metadata_path)
        self.fpath = folder_path
        self.fid = fid
        self.key = key
        self.caption_col = caption_col
        self.p = p
        self.transform = transform
        
    def __getitem__(self, idx):
        fid = self.df[self.fid][idx]
        key = self.df[self.key][idx]
        img_path = f"{self.fpath}/{fid}/{key}.jpg"
        img = Image.open(img_path).convert('RGB')
        
        prompts = []
        for col in self.caption_col:
            prompts.append(self.df[col][idx])
        caption = np.random.choice(prompts, p=self.p)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, caption
    
    def __len__(self):
        return len(self.df)
  
    
class ImageNet:
    def __init__(self, root, split='train', transform=None):
        self.dataset = torchvision.datasets.ImageNet(root=root, split=split)
        self.transform = transform
        self.prefix = ["an image of ", "a picture of "]
        
    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        caption = np.random.choice(self.prefix) + np.random.choice(self.dataset.classes[target])
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, caption
    
    def __len__(self):
        return len(self.dataset)
    

class EchoNet:
    def __init__(self, root_dir, image_list, split='train', transform=None):
        self.dataset = pd.read_csv(image_list)
        self.root_dir = root_dir
        self.transform = transform
        self.default_caption = "An echocardiogram image."
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.dataset.iloc[idx, 0])
        img = Image.open(img_path)
        caption = self.default_caption
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, caption
    