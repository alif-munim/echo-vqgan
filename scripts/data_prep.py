import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import pandas as pd

root_dir = '/scratch/alif/EchoNet-Dynamic/'
data_dir = os.path.join(root_dir, 'Images')


files = [f for f in tqdm(listdir(data_dir))]
print(len(files))
df = pd.DataFrame(files, columns=["image_path"])

save_path = os.path.join(root_dir, 'image_list.csv')
df.to_csv(save_path, index=False)

image_list = pd.read_csv(save_path)

print("Sampling first ten paths...")
for idx in range(10):
    img_path = os.path.join(data_dir, image_list.iloc[idx, 0])
    print(img_path)