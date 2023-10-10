import vqgan as vq
from PIL import Image
import torch
import numpy as np

img_path='og1.jpg'

img = Image.open(img_path).convert('RGB')
img = vq.stage1_transform(is_train=False)(img)

# load pretrained vit-vqgan
model = vq.create_model(arch='vqgan', version='vit-s-vqgan', pretrained=True)
# encode image to latent
z, _, _ = model.encode(img.unsqueeze(0))
# decode latent to image
rec = model.decode(z).squeeze(0)
rec = torch.clamp(rec, -1., 1.)

x = (rec + 1) * 0.5
x = x.permute(1,2,0).detach().cpu().numpy()
x = (255*x).astype(np.uint8)
x = Image.fromarray(x)

x.save("rec1.jpg")
