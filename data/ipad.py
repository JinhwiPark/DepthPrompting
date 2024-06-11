import os
import warnings
import numpy as np
import json
import h5py

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
import glob
import copy

warnings.filterwarnings("ignore", category=UserWarning)

class BaseDataset(Dataset):
    def __init__(self, args, mode):
        self.args = args

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    class ToNumpy:
        def __call__(self, sample):
            return np.array(sample)

class iPad(BaseDataset):
    def __init__(self, args, mode, num_sample_test=None):
        super(iPad, self).__init__(args, mode)

        self.args = args
        self.mode = mode
        self.num_sample = num_sample_test

        self.height = args.patch_height
        self.width = args.patch_width
        
        ipad_img_height = 1920
        ipad_img_width = 1440
        ipad_dep_height = 256
        ipad_dep_width = 192
        
        self.crop_size_img = (ipad_img_width*(self.height/self.width),ipad_img_width)
        self.crop_size_dep = (ipad_dep_width*(self.height/self.width),ipad_dep_width)

        self.sample_list = glob.glob('{}/*/*.npz'.format(self.args.dir_data))
        if len(self.sample_list)==0:
            self.sample_list = glob.glob('{}*/*.npz'.format(self.args.dir_data))
        print("LIST of iPad Bundle", self.sample_list)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        path_bundle = self.sample_list[idx]
        bundle = np.load(path_bundle, allow_pickle=True)
        i=5
        dep=bundle[f'depth_{i}'] 
        intrinsic=bundle[f'info_{i}'].item()['intrinsics']
        rgb=bundle[f'img_{i}']
        conf=bundle[f'conf_{i}']
        
        dep_=copy.deepcopy(dep)

        if self.args.conf_select:
            dep_[conf==1]=0.
            dep_[conf==0]=0.

        rgb = Image.fromarray(rgb, mode='RGB')
        dep = Image.fromarray(dep.astype('float32'), mode='F')
        dep_ = Image.fromarray(dep_.astype('float32'), mode='F')

        t_rgb = T.Compose([
            T.CenterCrop(self.crop_size_img),
            T.Resize(self.height),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        t_dep = T.Compose([
            T.CenterCrop(self.crop_size_dep),
            T.Resize(self.height),
            self.ToNumpy(),
            T.ToTensor()
        ])

        rgb = t_rgb(rgb)
        dep = t_dep(dep)
        dep_ = t_dep(dep_)
        
        dep_sp,ns = self.get_sparse_depth(dep_, self.num_sample, test=True)
        output = {'rgb': rgb, 'dep': dep_sp, 'gt': dep, 'K': intrinsic, 'num_sample':ns}
        return output

    def get_sparse_depth(self, dep, num_sample, test=False):
        channel, height, width = dep.shape

        assert channel == 1
        idx_nnz = torch.nonzero((dep.view(-1) > 0.0001) & (dep.view(-1) < 2.0), as_tuple=False)
        num_idx = len(idx_nnz)

        if test:
            idx_sample = torch.randperm(num_idx)[:num_sample]

        else:
            if num_sample == 'random' or num_sample=='random_high500':
                if num_sample == 'random_high500':
                    if random.randint(1, 2) == 2:
                        num_sample = 500
                    else:
                        num_sample = random.randint(1, 500)
                else:
                    num_sample = random.randint(1, 500)
                
            else:
                num_sample = int(num_sample)
            idx_sample = torch.randperm(num_idx)[:num_sample]           
        

        idx_nnz = idx_nnz[idx_sample[:]]

        mask = torch.zeros((channel*height*width))
        mask[idx_nnz] = 1.0
        mask = mask.view((channel, height, width))
        dep_sp = dep * mask.type_as(dep)
        return dep_sp, num_sample
    
    def get_specific_depth(self, dep, num_sample=None, type='grid', test=False):
        channel, height, width = dep.shape
        grid_length = 160
        start_point_h = 40
        start_point_w = 80
        num_sample = 16

        assert channel == 1

        mask = torch.zeros((channel*height*width))
        mask = mask.view((channel, height, width))
        mask[:, start_point_h:start_point_h+grid_length:30, start_point_w:start_point_w+grid_length:30] = 1.0

        dep_sp = dep * mask.type_as(dep)

        return dep_sp, num_sample