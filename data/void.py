import os
import warnings
import numpy as np
import json
import h5py
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import glob

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

class VOID(BaseDataset):
    def __init__(self, args, mode, num_sample_test=None):
        super(VOID, self).__init__(args, mode)

        self.args = args
        self.void_sparsity = self.args.void_sparsity
        self.data_path = self.args.dir_data
        self.mode = 'test'
        self.sample_list_rgb = glob.glob('{}/void_{}-*/*/image/*.png'.format(self.data_path, self.void_sparsity))
        self.sample_list_dep = glob.glob('{}/void_{}-*/*/sparse_depth/*.png'.format(self.data_path, self.void_sparsity))
        self.sample_list_gt = glob.glob('{}/void_{}-*/*/ground_truth/*.png'.format(self.data_path, self.void_sparsity))

        if len(self.sample_list_rgb)==0:
            self.sample_list_rgb = glob.glob('{}void_{}-*/*/image/*.png'.format(self.data_path, self.void_sparsity))
            self.sample_list_dep = glob.glob('{}void_{}-*/*/sparse_depth/*.png'.format(self.data_path, self.void_sparsity))
            self.sample_list_gt = glob.glob('{}void_{}-*/*/ground_truth/*.png'.format(self.data_path, self.void_sparsity))

        if args.patch_height is None:
            args.patch_height = 240
            args.patch_width = 320

        self.height = args.patch_height
        self.width = args.patch_width

        if self.height==240:
            self.crop_size = (228,304)
        elif self.height==480:
            self.crop_size = (456,608)
        elif self.height==256:
            self.crop_size = (256,192)
        else:
            raise print("Check the self.height !!")

    def __len__(self):
        return len(self.sample_list_rgb)

    def __getitem__(self, idx):
        rgb_path = self.sample_list_rgb[idx]
        dep_path = self.sample_list_dep[idx]
        gt_path = self.sample_list_gt[idx]
        rgb = Image.open(rgb_path)
        dep = Image.open(dep_path)
        gt = Image.open(gt_path)

        dep = np.array(dep).astype(np.float32)
        dep = dep / 1000.0
        dep = Image.fromarray(dep.astype('float32'), mode='F')

        gt = np.array(gt).astype(np.float32)
        gt = gt / 1000.0
        gt = Image.fromarray(gt.astype('float32'), mode='F')
        dep = dep.resize((240, 320),Image.NEAREST)
        gt = gt.resize((240, 320),Image.NEAREST)

        t_rgb = T.Compose([
            T.Resize(self.height),
            T.CenterCrop(self.crop_size), 
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        t_dep = T.Compose([
            T.CenterCrop(self.crop_size),
            self.ToNumpy(),
            T.ToTensor()
        ])
        rgb = t_rgb(rgb)
        dep = t_dep(dep)
        gt = t_dep(gt)
        output = {'rgb': rgb, 'dep': dep, 'gt': gt, 'K': dep}
        return output