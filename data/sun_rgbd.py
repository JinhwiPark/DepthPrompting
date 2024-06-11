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
from scipy import io

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

class SUN_RGBD(BaseDataset):
    def __init__(self, args, mode, num_sample_test=None):
        super(SUN_RGBD, self).__init__(args, mode)

        self.args = args
        self.mode = mode
        self.num_sample_test = num_sample_test
        self.use_raw_depth_as_input = self.args.use_raw_depth_as_input
        self.each_sensor = self.args.sun_rgbd_each_sensor

        if self.mode =='test':
            assert type(self.num_sample_test) is int, "TEST dataset should have specific # of sample !!"

        if mode != 'train' and mode != 'val' and mode != 'test':
            raise NotImplementedError

        self.height = args.patch_height 
        self.width = args.patch_width
        self.crop_size = (228, 304)

        split_file = io.loadmat(self.args.split_json)
        self.sample_list = split_file['alltest'][0]
        self.augment = False if self.mode == 'test' else self.args.augment

    def __len__(self):
        if not self.each_sensor:
            return len(self.sample_list)
        else:
            return len(self.sample_list_dep)

    def __getitem__(self, idx):
        path_default = self.args.dir_data.replace('/SUNRGBD', '/')
        rgb_path = path_default + os.path.join(self.sample_list[idx].tolist()[0][16:], 'image')
        rgb_path = glob.glob(rgb_path+'/*.jpg')[0]
        dep_path = path_default + os.path.join(self.sample_list[idx].tolist()[0][16:], 'depth')
        dep_path = glob.glob(dep_path+'/*.png')[0]
        gt_path = path_default + os.path.join(self.sample_list[idx].tolist()[0][16:], 'depth_bfx')
        gt_path = glob.glob(gt_path+'/*.png')[0]

        rgb = Image.open(rgb_path)
        dep = Image.open(dep_path)
        gt = Image.open(gt_path)    

        dep = np.array(dep).astype(np.float32)
        dep = dep / 10000.0  
        dep = Image.fromarray(dep.astype('float32'), mode='F')

        gt = np.array(gt).astype(np.float32)
        gt = gt / 10000.0  
        gt = Image.fromarray(gt.astype('float32'), mode='F')

        t_rgb = T.Compose([
            T.Resize(self.height),
            T.CenterCrop(self.crop_size), 
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        t_dep = T.Compose([
            T.Resize(self.height), 
            T.CenterCrop(self.crop_size), 
            self.ToNumpy(),
            T.ToTensor()
        ])
            
        rgb = t_rgb(rgb)
        dep = t_dep(dep)
        gt = t_dep(gt)

        dep_sp, ns = dep, self.num_sample_test

        output = {'rgb': rgb, 'dep': dep_sp, 'gt': gt, 'K': 0, 'rgb_480640':0, 'dep_480640':0, 'num_sample':ns}

        return output

    def get_sparse_depth(self, dep, num_sample, test=False, max_=500):
        channel, height, width = dep.shape

        assert channel == 1

        if self.args.pattern == 'grid':
            h=torch.arange(5,228,10)
            w=torch.arange(5,304,14)
            mask_h = torch.zeros(dep.shape)
            mask_w = torch.zeros(dep.shape)
            mask_h[:,h,:]=1.
            mask_w[:,:,w]=1.
            dep = dep * mask_h * mask_w
            return dep, dep.nonzero().shape[0]
        
        if self.args.range == 'under3':
            mask = torch.logical_and(dep > 1e-3, dep < 3)
            idx_nnz = torch.nonzero(mask.view(-1), as_tuple=False)
        elif self.args.range == 'over3':
            idx_nnz = torch.nonzero(dep.view(-1)> 3.0, as_tuple=False)
        else:
            idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)
        num_idx = len(idx_nnz)

        if test:
            g_cpu = torch.Generator()
            g_cpu.manual_seed(self.args.sample_seed)
            idx_sample = torch.randperm(num_idx, generator=g_cpu)[:num_sample]
        else:
            if num_sample == 'random' or num_sample=='random_high500':
                if num_sample == 'random_high500':
                    if random.randint(1, 2) == 2:
                        num_sample = 500
                    else:
                        num_sample = random.randint(1, 500)
                else:
                    num_sample = random.randint(1, max_)
                
            else:
                num_sample = int(num_sample)
            idx_sample = torch.randperm(num_idx)[:num_sample]           
        

        idx_nnz = idx_nnz[idx_sample[:]]

        mask = torch.zeros((channel*height*width))
        mask[idx_nnz] = 1.0
        mask = mask.view((channel, height, width))

        dep_sp = dep * mask.type_as(dep)
        
        return dep_sp, num_sample


    def get_sparse_depth_test(self, dep, num_sample):
        channel, height, width = dep.shape
        assert channel == 1

        if self.args.exp_name == 'PatternChange': 

            h=torch.arange(5,228,10)
            w=torch.arange(5,304,14)
            mask_h = torch.zeros(dep.shape)
            mask_w = torch.zeros(dep.shape)
            mask_h[:,h,:]=1.
            mask_w[:,:,w]=1.
            dep = dep * mask_h * mask_w
            return dep, dep.nonzero().shape[0]
        
        if self.args.exp_name == 'RangeChange-Near': 
            idx_nnz = torch.nonzero(dep.view(-1)> 3.0, as_tuple=False)
        elif self.args.exp_name == 'RangeChange-Far': 
            mask = torch.logical_and(dep > 1e-3, dep < 3)
            idx_nnz = torch.nonzero(mask.view(-1), as_tuple=False)
        else:
            idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)
        num_idx = len(idx_nnz)

        g_cpu = torch.Generator()
        g_cpu.manual_seed(self.args.sample_seed)
        idx_sample = torch.randperm(num_idx, generator=g_cpu)[:num_sample]       
    
        idx_nnz = idx_nnz[idx_sample[:]]

        mask = torch.zeros((channel*height*width))
        mask[idx_nnz] = 1.0
        mask = mask.view((channel, height, width))

        dep_sp = dep * mask.type_as(dep)
        return dep_sp, num_sample