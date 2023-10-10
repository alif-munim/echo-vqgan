import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange

x = torch.rand(16, 3, 256, 256)
y = torch.rand(16, 3, 112, 112)

in_channels = 3
dim = 512
image_size = 256
patch_size = 8
num_patches = (image_size // patch_size) ** 2
scale = dim ** -0.5

to_patch_embedding = nn.Sequential(
    nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size, bias=False),
    Rearrange('b c h w -> b (h w) c'),
)

position_embedding = nn.Parameter(torch.randn(1, num_patches, dim) * scale)

x_patch = to_patch_embedding(x)
print("\nx: ", x.shape)
print("x_patch: ", x_patch.shape)
print("pos_emb: ", position_embedding.shape)

x_res = x_patch + position_embedding
print(x_res.shape)


to_patch_embedding = nn.Sequential(
    nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size, bias=False),
    Rearrange('b c h w -> b (h w) c'),
)

y_patch = to_patch_embedding(y)
print("\ny: ", y.shape)
print("y_patch: ", y_patch.shape)
print("pos_emb: ", position_embedding.shape)
print()

y_res = y_patch + position_embedding
print(y_res.shape)
