import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
import logging

# from datasets.imagenet import get_imagenet_dataset
# from utils.degredations import H_functions, Inpainting, Inpainting2


logging.basicConfig(level=logging.INFO)

def random_sq_bbox(img, mask_shape, image_size=256, margin=(16, 16)):
    """Generate a random sqaure mask for inpainting
    """
    B, C, H, W = img.shape
    h, w = mask_shape
    margin_height, margin_width = margin
    maxt = image_size - margin_height - h
    maxl = image_size - margin_width - w

    # bb
    t = np.random.randint(margin_height, maxt)
    l = np.random.randint(margin_width, maxl)

    # make mask
    mask = torch.ones([B, C, H, W], device=img.device)
    mask[..., t:t+h, l:l+w] = 0

    return mask, t, t+h, l, l+w


def random_mask(img, mask_shape, image_size=256, margin=(16, 16)):
    """Generate a random sqaure mask for inpainting
    """
    B, C, H, W = img.shape
    h, w = mask_shape
    margin_height, margin_width = margin
    maxt = image_size - margin_height - h
    maxl = image_size - margin_width - w

    # bb
    t = np.random.randint(margin_height, maxt)
    l = np.random.randint(margin_width, maxl)

    # make mask
    mask = torch.ones([B, C, H, W], device=img.device)
    mask[..., t:t+h, l:l+w] = 0

    return mask, t, t+h, l, l+w


def save_mask(f, mask):
    m_npy = mask.astype(bool)
    m = np.packbits(m_npy, axis=None)
    shape = m_npy.shape
    np.savez(f, m=m, shape=shape)

def main():
    img = torch.randn(1, 3, 256, 256)
    mask = torch.cat([random_sq_bbox(img, [128, 128])[0].view(1, -1) for i in range(1000)], 
                     dim=0)
    # check if the directory exists 
    if not os.path.exists('<root>/_exp/masks'):
        os.makedirs('<root>/_exp/masks')
        
    # check overwriting
    if os.path.exists('<root>/_exp/masks/20ff.npz'):
        logging.info("Removing existing mask")
        os.remove('<root>/_exp/masks/20ff.npz')
    save_mask('<root>/_exp/masks/20ff.npz', mask.cpu().numpy())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Generating inpainting masks")
    main()
    logging.info("Done")



