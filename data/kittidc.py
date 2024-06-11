import os
import numpy as np
import json
import random
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import cv2

from .lidar import sample_lidar_lines

MEAN=(123.675, 116.28, 103.53)
STD=(58.395, 57.12, 57.375)

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

def read_depth(file_name):
    assert os.path.exists(file_name), "file not found: {}".format(file_name)
    image_depth = np.array(Image.open(file_name))

    assert (np.max(image_depth) == 0) or (np.max(image_depth) > 255), \
        "np.max(depth_png)={}, path={}".format(np.max(image_depth), file_name)

    image_depth = image_depth.astype(np.float32) / 256.0
    return image_depth

def read_calib_file(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

class KITTIDC(BaseDataset):
    def __init__(self, args, mode, num_lidars_test=None):
        super(KITTIDC, self).__init__(args, mode)

        self.args = args
        self.mode = mode
        self.val_lidar_lines = num_lidars_test

        if mode =='test':
            assert type(self.val_lidar_lines) is int, "TEST dataset should have specifict # of sample !!"

        if mode != 'train' and mode != 'val' and mode != 'test':
            raise NotImplementedError

        self.height = args.patch_height
        self.width = args.patch_width
        self.augment = self.args.augment

        with open(self.args.split_json) as json_file:
            json_data = json.load(json_file)
            self.sample_list = json_data[mode]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        rgb, depth, gt, K = self._load_data(idx)
        rgb_ori=0

        if self.augment and self.mode == 'train':
            if self.args.top_crop > 0:
                width, height = rgb.size
                rgb = TF.crop(rgb, self.args.top_crop, 0,
                              height - self.args.top_crop, width)                   
                depth = TF.crop(depth, self.args.top_crop, 0,
                                height - self.args.top_crop, width)
                gt = TF.crop(gt, self.args.top_crop, 0,
                             height - self.args.top_crop, width)
                K[3] = K[3] - self.args.top_crop

            width, height = rgb.size

            _scale = np.random.uniform(1.0, 1.5)
            scale = int(height * _scale)
            degree = np.random.uniform(-5.0, 5.0)
            flip = np.random.uniform(0.0, 1.0)

            if flip > 0.5:
                rgb = TF.hflip(rgb)
                depth = TF.hflip(depth)
                gt = TF.hflip(gt)
                K[2] = width - K[2]

            rgb = TF.rotate(rgb, angle=degree, resample=Image.BICUBIC)
            depth = TF.rotate(depth, angle=degree, resample=Image.NEAREST)
            gt = TF.rotate(gt, angle=degree, resample=Image.NEAREST)

            if self.args._kittiori:
                rgb_ori = rgb
            
            brightness = np.random.uniform(0.6, 1.4)
            contrast = np.random.uniform(0.6, 1.4)
            saturation = np.random.uniform(0.6, 1.4)

            rgb = TF.adjust_brightness(rgb, brightness)
            rgb = TF.adjust_contrast(rgb, contrast)
            rgb = TF.adjust_saturation(rgb, saturation)

            rgb = TF.resize(rgb, scale, Image.BICUBIC)
            if self.args._kittiori:
                rgb_ori = TF.resize(rgb_ori, scale, Image.BICUBIC)
            depth = TF.resize(depth, scale, Image.NEAREST)
            gt = TF.resize(gt, scale, Image.NEAREST)

            K[0] = K[0] * _scale
            K[1] = K[1] * _scale
            K[2] = K[2] * _scale
            K[3] = K[3] * _scale

            width, height = rgb.size

            assert self.height <= height and self.width <= width, \
                "patch size is larger than the input size"

            h_start = random.randint(0, height - self.height)
            w_start = random.randint(0, width - self.width)

            rgb = TF.crop(rgb, h_start, w_start, self.height, self.width)
            if self.args._kittiori:
                rgb_ori = TF.crop(rgb_ori, h_start, w_start, self.height, self.width)
            depth = TF.crop(depth, h_start, w_start, self.height, self.width)
            gt = TF.crop(gt, h_start, w_start, self.height, self.width)

            K[2] = K[2] - w_start
            K[3] = K[3] - h_start

            rgb = TF.to_tensor(rgb)
            rgb = TF.normalize(rgb, (0.485, 0.456, 0.406),(0.229, 0.224, 0.225), inplace=True)
            if self.args._kittiori:
                rgb_ori = TF.to_tensor(rgb_ori)
                rgb_ori = TF.normalize(rgb_ori, (0.485, 0.456, 0.406),(0.229, 0.224, 0.225), inplace=True)

            depth = TF.to_tensor(np.array(depth))
            depth = depth / _scale

            gt = TF.to_tensor(np.array(gt))
            gt = gt / _scale

        elif self.mode in ['train', 'val']:
            if self.args.top_crop > 0:
                width, height = rgb.size
                rgb = TF.crop(rgb, self.args.top_crop, 0,
                              height - self.args.top_crop, width)
                depth = TF.crop(depth, self.args.top_crop, 0,
                                height - self.args.top_crop, width)
                gt = TF.crop(gt, self.args.top_crop, 0,
                             height - self.args.top_crop, width)
                K[3] = K[3] - self.args.top_crop

        
            width, height = rgb.size

            assert self.height <= height and self.width <= width, \
                "patch size is larger than the input size"

            h_start = random.randint(0, height - self.height)
            w_start = random.randint(0, width - self.width)

            rgb = TF.crop(rgb, h_start, w_start, self.height, self.width)
            depth = TF.crop(depth, h_start, w_start, self.height, self.width)
            gt = TF.crop(gt, h_start, w_start, self.height, self.width)

            K[2] = K[2] - w_start
            K[3] = K[3] - h_start

            rgb = TF.to_tensor(rgb)
            rgb = TF.normalize(rgb, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
            if self.args._kittiori:
                rgb_ori = rgb

            depth = TF.to_tensor(np.array(depth))

            gt = TF.to_tensor(np.array(gt))
        else:
            if self.args.top_crop > 0 and self.args.top_crop:
                width, height = rgb.size
                rgb = TF.crop(rgb, self.args.top_crop, 0,
                              height - self.args.top_crop, width)
                depth = TF.crop(depth, self.args.top_crop, 0,
                                height - self.args.top_crop, width)
                gt = TF.crop(gt, self.args.top_crop, 0,
                             height - self.args.top_crop, width)
                K[3] = K[3] - self.args.top_crop

                width, height = rgb.size

            rgb = TF.to_tensor(rgb)
            rgb = TF.normalize(rgb, (0.485, 0.456, 0.406),(0.229, 0.224, 0.225), inplace=True)

            depth = TF.to_tensor(np.array(depth))

            gt = TF.to_tensor(np.array(gt))
            if self.args._kittiori:
                rgb_ori = rgb

        if self.args.num_sample == None:
            self.args.num_sample = 0
        if self.args.num_sample > 0:
            print(" Sampling for the sparse depth",self.args.num_sample)
            
            print(gt.nonzero().shape)
            depth = self.get_sparse_depth(gt, self.args.num_sample)
            print(depth.nonzero().shape)
            
        output = {'rgb': rgb, 'dep': depth, 'gt': gt, 'K': torch.Tensor(K), 'rgb_ori': rgb_ori}

        return output

    def _load_data(self, idx):

        path_rgb = os.path.join(self.args.dir_data,
                                self.sample_list[idx]['rgb'])
        path_depth = os.path.join(self.args.dir_data,
                                  self.sample_list[idx]['depth'])
        path_gt = os.path.join(self.args.dir_data,
                               self.sample_list[idx]['gt'])
        path_calib = os.path.join(self.args.dir_data,
                                  self.sample_list[idx]['K'])

        depth = read_depth(path_depth)
        gt = read_depth(path_gt)

        
        if self.mode in ['train', 'val']:
            calib = read_calib_file(path_calib)    
            if 'image_02' in path_rgb:
                K_cam = np.reshape(calib['P_rect_02'], (3, 4))
            elif 'image_03' in path_rgb:
                K_cam = np.reshape(calib['P_rect_03'], (3, 4))
            K = [K_cam[0, 0], K_cam[1, 1], K_cam[0, 2], K_cam[1, 2]]
        else:
            f_calib = open(path_calib, 'r')
            K_cam = f_calib.readline().split(' ')
            f_calib.close()
            K = [float(K_cam[0]), float(K_cam[4]), float(K_cam[2]),
                 float(K_cam[5])]
            
        if self.args.lidar_lines=='1' or self.args.lidar_lines=='4' or self.args.lidar_lines=='16':
            keep_ratio = (float(self.args.lidar_lines) / 64.0)
            Km = np.eye(3)
            Km[0, 0] = K[0]
            Km[1, 1] = K[1]
            Km[0, 2] = K[2]
            Km[1, 2] = K[3]
            depth = sample_lidar_lines(depth[:, :, None], intrinsics=Km, keep_ratio=keep_ratio)[:, :, 0]

        elif (self.args.lidar_lines=='random_lidar' and self.mode =='train'):
            if self.args.lidar_lines=='random_lidar':
                self.lidar_lines= random.randint(1, 64)
            keep_ratio = (self.lidar_lines / 64.0)
            assert keep_ratio >=0 and keep_ratio <= 1.0, keep_ratio
            if keep_ratio >= 0.9999:
                pass
            elif keep_ratio > 0:
                Km = np.eye(3)
                Km[0, 0] = K[0]
                Km[1, 1] = K[1]
                Km[0, 2] = K[2]
                Km[1, 2] = K[3]
                depth = sample_lidar_lines(depth[:, :, None], intrinsics=Km, keep_ratio=keep_ratio)[:, :, 0]
            else:
                depth = np.zeros_like(depth)

        elif self.mode == 'test' or self.mode == 'val':
            keep_ratio = (self.val_lidar_lines / 64.0)
            assert keep_ratio >=0 and keep_ratio <= 1.0, keep_ratio
            if keep_ratio >= 0.9999:
                pass
            elif keep_ratio > 0:
                Km = np.eye(3)
                Km[0, 0] = K[0]
                Km[1, 1] = K[1]
                Km[0, 2] = K[2]
                Km[1, 2] = K[3]
                depth = sample_lidar_lines(depth[:, :, None], intrinsics=Km, keep_ratio=keep_ratio)[:, :, 0]
            else:
                depth = np.zeros_like(depth)

        rgb = Image.open(path_rgb)
        depth = Image.fromarray(depth.astype('float32'), mode='F')
        gt = Image.fromarray(gt.astype('float32'), mode='F')

        w1, h1 = rgb.size
        w2, h2 = depth.size
        w3, h3 = gt.size

        assert w1 == w2 and w1 == w3 and h1 == h2 and h1 == h3

        return rgb, depth, gt, K

    def get_sparse_depth(self, dep, num_sample):
        channel, height, width = dep.shape

        assert channel == 1

        idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)

        num_idx = len(idx_nnz)
        idx_sample = torch.randperm(num_idx)[:num_sample]

        idx_nnz = idx_nnz[idx_sample[:]]

        mask = torch.zeros((channel*height*width))
        mask[idx_nnz] = 1.0
        mask = mask.view((channel, height, width))

        dep_sp = dep * mask.type_as(dep)

        return dep_sp

def imnormalize(img, mean, std, to_rgb=True):
    img = np.float32(img)
    return imnormalize_(img, mean, std, to_rgb)

def imnormalize_(img, mean, std, to_rgb=True):
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img) 
    cv2.subtract(img, mean, img) 
    cv2.multiply(img, stdinv, img) 
    return img



