import os
import torch
import zipfile
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
from datasets import load_dataset
from skimage.util import random_noise
import random

    
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
        self.gauss_var = 0.09
        self.speckle_var = 0.5625
        self.sp_amount = 0.001
        
        self.dataset = pd.read_csv(image_list, 
                                   nrows=1000 # take a smaller chunk to test epoch and inplace operations
        )
        self.root_dir = root_dir
        self.transform = transform
        self.default_caption = "An echocardiogram image."
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        gauss_var = random.uniform(0.00, 0.0225)
        speckle_var = random.uniform(0.00, 0.5625)
        sp_amount = random.uniform(0.00, 0.02)
        
        img_path = os.path.join(self.root_dir, self.dataset.iloc[idx, 0])
        img = Image.open(img_path)
        
        noised_arr = np.asarray(img)
        noised_arr = random_noise(noised_arr, mode='gaussian', var=gauss_var)
        noised_arr = random_noise(noised_arr, mode='speckle', var=speckle_var)
        noised_arr = random_noise(noised_arr, mode='s&p', amount=sp_amount)
        noised_img = Image.fromarray((noised_arr*255).astype(np.uint8))
        
        noised_img = noised_img.convert('L')
        img = img.convert('L')
        
        # caption = self.default_caption
        
        # Debug statements
        # noised_img.save(f'/scratch/alif/echo-vqgan/test/noisy_{self.dataset.iloc[idx, 0]}.jpg')
        # img.save(f'/scratch/alif/echo-vqgan/test/clear_{self.dataset.iloc[idx, 0]}.jpg')
        # print(f'Saved {self.dataset.iloc[idx, 0]} noisy, clean images.')
        
        if self.transform is not None:
            img = self.transform(img)
            noised_img = self.transform(noised_img)
            
        return img, noised_img
    