import os
import sys
sys.path.insert(0, '..')

import vqgan as vq
from vqgan.utils import datasets
import torch

root_dir = '/scratch/alif/EchoNet-Dynamic/'
data_path = os.path.join(root_dir, 'Images')
annotations = os.path.join(root_dir, 'image_list.csv')
mode = "grayscale"

vqvae_checkpoint = "/scratch/alif/echo-vqgan/scripts/results/models/grayscale_vqvae_step_41580.pt"
discr_checkpoint = "/scratch/alif/echo-vqgan/scripts/results/models/grayscale_discr_step_41580.pt"

transform = vq.stage1_transform(img_size=112, is_train=True, scale=0.66)
dataset = datasets.EchoNet(
            root_dir=data_path, 
            image_list=annotations, 
            mode=mode,
            transform=transform
)

version = 'vit-s-vqgan-' + mode
model = vq.create_model(arch='vqgan', version=version, pretrained=False)

trainer = vq.VQGANTrainer(
    vqvae                    = model,
    dataset                  = dataset,
    num_epoch                = 100,
    valid_size               = 64,
    lr                       = 1e-4,
    lr_min                   = 5e-5,
    warmup_steps             = 50000,
    warmup_lr_init           = 1e-6,
    decay_steps              = 100000,
    batch_size               = 128,
    num_workers              = 2,
    pin_memory               = True,
    grad_accum_steps         = 8,
    mixed_precision          = 'no',
    max_grad_norm            = 1.0,
    save_every               = 330,
    sample_every             = 330,
    result_folder            = "results",
    log_dir                  = "logs",
    mode                     = mode,
    vqvae_checkpoint         = vqvae_checkpoint,
    discr_checkpoint         = discr_checkpoint
)

trainer.train()