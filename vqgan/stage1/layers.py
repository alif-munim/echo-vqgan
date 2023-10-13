import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
from ..modules.attention import CrossAttention, MemoryEfficientCrossAttention, XFORMERS_IS_AVAILBLE
from ..modules.mlp import SwiGLUFFNFused


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.):
        super().__init__()
        self.w_1 = nn.Linear(dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.w_2 = nn.Linear(mlp_dim, dim)
    
    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)

        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Layer(nn.Module):
    ATTENTION_MODES = {
        "vanilla": CrossAttention,
        "xformer": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, dim_head, mlp_dim, num_head=8, dropout=0.0):
        super().__init__()
        attn_mode = "xformer" if XFORMERS_IS_AVAILBLE else "vanilla"
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = attn_cls(query_dim=dim, heads=num_head, dim_head=dim_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffnet = SwiGLUFFNFused(in_features=dim, hidden_features=mlp_dim)
        
    def forward(self, x): 
        # print(f'LAYER: initial x nan: {torch.isnan(x).any()}, initial x inf: {torch.isinf(x).any()}')
        x = self.attn1(self.norm1(x)) + x
        # print(f'LAYER: after attn1, norm1 nan: {torch.isnan(x).any()}, after attn1, norm1 inf: {torch.isinf(x).any()}')
        x = self.ffnet(self.norm2(x)) + x
        # print(f'LAYER: after ffnet, norm2 nan: {torch.isnan(x).any()}, after ffnet, norm2 inf: {torch.isinf(x).any()}')
        
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, num_head, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.Sequential(*[Layer(dim, dim_head, mlp_dim, num_head, dropout) for i in range(depth)])
    
    def forward(self, x):
        x = self.layers(x)
        
        return x


class Encoder(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, num_head, mlp_dim, in_channels=3, out_channels=3, dim_head=64, dropout=0.):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size, bias=False),
            Rearrange('b c h w -> b (h w) c'),
        )
        
        scale = dim ** -0.5
        num_patches = (image_size // patch_size) ** 2
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches, dim) * scale)
        self.norm_pre = nn.LayerNorm(dim)
        self.transformer = Transformer(dim, depth, num_head, dim_head, mlp_dim, dropout)
        
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # print(f'ENCODER: initial x nan: {torch.isnan(x).any()}, encoder x inf: {torch.isinf(x).any()}')
        x = self.to_patch_embedding(x)
        x = x + self.position_embedding
        # print(f'ENCODER: after pos emb nan: {torch.isnan(x).any()}, after pos emb inf: {torch.isinf(x).any()}')
        x = self.norm_pre(x)
        # print(f'ENCODER: after norm nan: {torch.isnan(x).any()}, after norm inf: {torch.isinf(x).any()}')
        x = self.transformer(x)
        # print(f'ENCODER: after transformer nan: {torch.isnan(x).any()}, after transformer inf: {torch.isinf(x).any()}')
        
        return x
 
       
class Decoder(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, num_head, mlp_dim, in_channels=3, out_channels=3, dim_head=64, dropout=0.):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        scale = dim ** -0.5
        num_patches = (image_size // patch_size) ** 2
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches, dim) * scale)
        self.transformer = Transformer(dim, depth, num_head, dim_head, mlp_dim, dropout)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, out_channels * patch_size * patch_size, bias=True)
        
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        # print(f'DECODER: initial x nan: {torch.isnan(x).any()}, decoder x inf: {torch.isinf(x).any()}')
        x = x + self.position_embedding
        # print(f'DECODER: after pos emb nan: {torch.isnan(x).any()}, after pos emb inf: {torch.isinf(x).any()}')
        x = self.transformer(x)
        # print(f'DECODER: after transformer nan: {torch.isnan(x).any()}, after transformer inf: {torch.isinf(x).any()}')
        x = self.norm(x)
        # print(f'DECODER: after norm nan: {torch.isnan(x).any()}, after norm inf: {torch.isinf(x).any()}')
        x = self.proj(x)
        # print(f'DECODER: after proj nan: {torch.isnan(x).any()}, after proj inf: {torch.isinf(x).any()}')
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=self.image_size//self.patch_size, p1=self.patch_size, p2=self.patch_size)
        
        return x

