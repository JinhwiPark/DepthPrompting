import os
import numpy as np
import json
import h5py
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import cv2

import sys
sys.path.append(os.path.join(os.getcwd(), 'data/nuscene_utils'))
from lib.dataset_processors import NuScenesProcessor
from datasets import NuScenesDataset

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class BaseDataset(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    class ToNumpy:
        def __call__(self, sample):
            return np.array(sample)

class NUSCENE(BaseDataset):
    def __init__(self, args, mode, num_sample_test=None):
        super(NUSCENE, self).__init__(args, mode)

        self.args = args
        self.mode = mode
        self.num_sample_test = num_sample_test
        if args.patch_height is None:
            args.patch_height = 900
            args.patch_width = 1600

        self.data_path = self.args.dir_data

        nusc_proc = NuScenesProcessor("v1.0-mini", self.data_path,
        [0, -1, 1], speed_bound=[0, np.inf],
        camera_channels=["CAM_FRONT"],
        pass_filters=['day', 'night', 'rain'],
        use_keyframe=True,
        stationary_filter=True,
        seg_mask='color', how_to_gen_masks='black',
        maskrcnn_batch_size=str(args.batch_size),
        regen_masks=True, subset_ratio=1.0)

        self.dataset = NuScenesDataset(
            self.data_path, nusc_proc.gen_tokens(is_train=True),
            args.patch_height, args.patch_width, [0, -1, 1], len([0, 1, 2, 3]),
            is_train=False, not_do_color_aug=True,
            not_do_flip=True, do_crop=False,
            crop_bound=True, seg_mask='color',
            boxify=False, MIN_OBJECT_AREA=20, 
            use_radar=False, use_lidar=True,
            prob_to_mask_objects=0.0,
            proc=nusc_proc)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        rgb = sample['color', 0, 0]
        dep_sp = sample['lidar', 0, 0].unsqueeze(0)
        dep = None
        K = None
        rgb_480640 = None
        dep_480640 = None
        ns = None
        dep_sp = dep_sp.float()

        normalize_only = T.Compose([
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        normalize_crop = T.Compose([
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            T.CenterCrop((352, 1600)),
            T.Resize((240, 1216), interpolation=Image.BILINEAR)
        ])

        center_crop_dep = T.Compose([
             T.CenterCrop((352, 1600)),
             T.Resize((240, 1216), interpolation=Image.NEAREST)
         ])   

        rgb = normalize_crop(rgb)
        dep_sp = center_crop_dep(dep_sp)


        output = {'rgb': rgb, 'dep': dep_sp, 'gt': dep_sp, 'K': dep_sp, 'rgb_480640':dep_sp, 'dep_480640':dep_sp, 'num_sample':dep_sp}
        return output




