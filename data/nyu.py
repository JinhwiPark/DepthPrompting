import os
import warnings
import random
import json
import h5py
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

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

class NYU(BaseDataset):
    def __init__(self, args, mode, num_sample_test=None):
        super(NYU, self).__init__(args, mode)

        self.args = args
        self.mode = mode
        self.num_sample = num_sample_test

        if self.mode =='test':
            assert type(self.num_sample) is int, "TEST dataset should have specific # of sample !!"

        if mode != 'train' and mode != 'val' and mode != 'test':
            raise NotImplementedError

        self.height = args.patch_height
        self.width = args.patch_width

        if self.height==240:
            self.crop_size = (228,304)
        elif self.height==480:
            self.crop_size = (456,608)
        else:
            raise print("Check the self.height !!")

        self.K = torch.Tensor([
            5.1885790117450188e+02 / 2.0,
            5.1946961112127485e+02 / 2.0,
            3.2558244941119034e+02 / 2.0 - 8.0,
            2.5373616633400465e+02 / 2.0 - 6.0
        ])
        self.augment = False if self.mode == 'test' else self.args.augment

        with open(self.args.split_json) as json_file:
            json_data = json.load(json_file)
            self.sample_list = json_data[mode]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        path_file = os.path.join(self.args.dir_data,
                                 self.sample_list[idx]['filename'])

        f = h5py.File(path_file, 'r')
        rgb_h5 = f['rgb'][:].transpose(1, 2, 0)
        dep_h5 = f['depth'][:]

        rgb = Image.fromarray(rgb_h5, mode='RGB')
        dep = Image.fromarray(dep_h5.astype('float32'), mode='F')
        
        rgb_480640 = 0
        dep_480640 = 0
        
        if self.mode == 'train' and self.augment:
            _scale = np.random.uniform(1.0, 1.5)
            scale = int(self.height * _scale)
            degree = np.random.uniform(-5.0, 5.0)
            flip = np.random.uniform(0.0, 1.0)

            if flip > 0.5:
                rgb = TF.hflip(rgb)
                dep = TF.hflip(dep)

            rgb = TF.rotate(rgb, angle=degree, resample=Image.NEAREST)
            dep = TF.rotate(dep, angle=degree, resample=Image.NEAREST)
                
            t_rgb = T.Compose([
                T.Resize(scale),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            t_dep = T.Compose([
                T.Resize(scale),
                T.CenterCrop(self.crop_size),
                self.ToNumpy(),
                T.ToTensor()
            ])

            if self.args._480640:
                rgb_480640=rgb
                t_rgb_480640 = T.Compose([
                                        T.Resize(scale),
                                        T.CenterCrop(self.crop_size),
                                        T.Resize((456,608)),
                                        T.ToTensor(),
                                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                        ])
                rgb_480640 = t_rgb_480640(rgb_480640)

            rgb = t_rgb(rgb)
            dep = t_dep(dep)

            dep = dep / _scale

            K = self.K.clone()
            K[0] = K[0] * _scale
            K[1] = K[1] * _scale
            
        else:
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

            if self.args._480640:
                rgb_480640=rgb
                t_rgb_480640 = T.Compose([
                                        T.Resize(480),
                                        T.CenterCrop((456,608)),
                                        T.ToTensor(),
                                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                        ])
                rgb_480640 = t_rgb_480640(rgb_480640)
                
                dep_480640=dep
                t_dep_480640 = T.Compose([
                                        T.Resize(480),
                                        T.CenterCrop((456,608)),
                                        self.ToNumpy(),
                                        T.ToTensor()
                                        ])
                dep_480640 = t_dep_480640(dep_480640)
                
                
            rgb = t_rgb(rgb)
            dep = t_dep(dep)
            K = self.K.clone()

        if self.mode =='test':
            dep_sp,ns = self.get_sparse_depth(dep, self.num_sample, test=True)
        else:
            dep_sp,ns = self.get_sparse_depth(dep, self.args.num_sample, test=False, max_=self.args.sp_max)

        output = {'rgb': rgb, 'dep': dep_sp, 'gt': dep, 'K': K, 'rgb_480640':rgb_480640, 'dep_480640':dep_480640, 'num_sample':ns}

        return output

    def get_sparse_depth(self, dep, num_sample, test=False, max_=500):
        channel, height, width = dep.shape

        assert channel == 1

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