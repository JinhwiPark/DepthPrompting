# Copyright © 2022, Bolian Chen. Released under the MIT license.

import os
from tqdm import tqdm
import numpy as np
import imageio
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn

import datasets

def generate_seg_masks(img_paths, threshold=0.5, seg_mask='color',
        batch_size=4, num_workers=4, regen_masks=False):
    """Generate segmentation masks of all the input images
    Args:
        img_paths(list): list of image paths to generate segmentation masks
    """
    model = maskrcnn_resnet50_fpn(pretrained=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    if not regen_masks:
        img_paths = [img for img in img_paths if not os.path.exists(
            os.path.splitext(img)[0] + '-fseg.png')]
    
    img_dataset = datasets.Paths2ImagesDataset(img_paths)
    img_dataloader = DataLoader(img_dataset,
                                batch_size = batch_size,
                                num_workers = num_workers)

    print("Generating segmentation masks with Mask R-CNN")
    for images, paths in tqdm(img_dataloader):
        images = list(images.to(device))
        with torch.no_grad():
            mrcnn_results = model(images)

        for i in range(len(images)):
            mrcnn_result = mrcnn_results[i]
            path = paths[i]

            # mask post-processing
            # only consider objects with predicted scores higher than threshold
            score = mrcnn_result['scores'].detach().cpu().numpy()
            valid = (score > threshold).sum()
            masks = (mrcnn_result['masks'] > threshold).squeeze(1).detach().cpu().numpy()
            labels = mrcnn_result['labels'].detach().cpu().numpy() 
            if valid > 0:
                masks = masks[:valid] # (N, H, W)
                labels = labels[:valid]
            else:
                masks = np.zeros_like(masks[:1])
                labels = np.zeros_like(labels[:1])
            masks = masks.astype(np.uint8)

            # Throw away the masks that are not pedestrians or vehicles
            masks[labels == 0] *= 0 # __background__
            masks[labels == 5] *= 0 # airplane
            masks[labels == 7] *= 0 # train
            masks[labels > 8] *= 0

            # Color ids for masks
            COLORS = np.arange(1, 256, dtype=np.uint8).reshape(-1, 1, 1)

            # TODO: self.mask
            mask_img = np.ones_like(masks, dtype=np.uint8) 
            if seg_mask == 'mono':
                mask_img = masks * mask_img
                mask_img = np.sum(mask_img, axis=0)
                mask_img = (mask_img > 0).astype(np.uint8) * 255
            elif seg_mask == 'color':
                for i in range(masks.shape[0]-1):
                    masks[i+1:] *= 1 - masks[i]
                # ignore this step when masks is empty 
                if masks.shape[0] != 0:
                    # for non-background objects
                    # sample colors evenly between 1 and 255
                    mask_img = masks * mask_img * COLORS[
                            np.linspace(0, 254, num= masks.shape[0], dtype=np.uint8)
                            ]
                mask_img = np.sum(mask_img, axis=0)

            mask_path = os.path.splitext(path)[0] + '-fseg.png'
            imageio.imsave(mask_path, mask_img.astype(np.uint8))


