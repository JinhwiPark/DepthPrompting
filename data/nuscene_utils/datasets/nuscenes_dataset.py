import os
import random
import bisect
import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt

from .mono_dataset import pil_loader, MonoDataset

import torch
import torch.utils.data as data
from torchvision import transforms


class NuScenesDataset(MonoDataset):
    
    def __init__(self, *args, **kwargs):

        super(NuScenesDataset, self).__init__(*args, **kwargs)
        self.nusc_proc = kwargs['proc']
        self.nusc = self.nusc_proc.get_nuscenes_obj()

    def check_depth(self):

        return False

    def __getitem__(self, index):

        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_color_aug = (not self.not_do_color_aug) and do_color_aug

        do_flip = self.is_train and random.random() > 0.5
        do_flip = (not self.not_do_flip) and do_flip

        if self.do_crop:
            crop_offset = -1 if self.is_train else -2
        else:
            crop_offset = -3

        token = self.filenames[index]

        for i in self.frame_idxs:
            inputs[('token', i)], color_info = self.get_color(
                    token, i, do_flip, crop_offset)
            inputs[('color', i, -1)], ratio, delta_u, delta_v, crop_offset = (
                    color_info)

            if self.seg_mask != 'none':
                mask = self.get_mask(token, i, do_flip, crop_offset)[0]
                inputs[('mask', i, -1)] = mask.convert('L')
                                                        

            if self.use_radar:
                inputs[('radar', i, 0)] = self.get_sensor_map(
                        inputs[('token', i)], ratio, delta_u, delta_v,
                        do_flip, sensor_type = 'radar')
            if self.use_lidar:
                inputs[('lidar', i, 0)] = self.get_sensor_map(
                        inputs[('token', i)], ratio, delta_u, delta_v,
                        do_flip, sensor_type = 'lidar')


        K = self.load_intrinsics(token)
        self.adjust_intrinsics(K, inputs, ratio, delta_u, delta_v, do_flip)
        
        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)
        

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[('token', i)]
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]
            if self.seg_mask != 'none':
                del inputs[("mask", i, -1)]

        return inputs

    def get_color(self, token, frame_id, do_flip, crop_offset=-3):
        token = self.nusc_proc.get_adjacent_token(token, frame_id)
        sample_data = self.nusc.get('sample_data', token)
        img_path = os.path.join(self.data_path, sample_data['filename'])
        return token, self.get_image(self.loader(img_path), do_flip, crop_offset)

    def get_mask(self, token, frame_id, do_flip, crop_offset=-3):

        token = self.nusc_proc.get_adjacent_token(token, frame_id)
        mask = self.nusc_proc.get_seg_mask(token)
        return self.get_image(mask, do_flip, crop_offset, inter=cv2.INTER_NEAREST)

    def load_intrinsics(self, token):

        K = self.nusc_proc.get_cam_intrinsics(token)
        K = np.concatenate( (K, np.array([[0,0,0]]).T), axis = 1 )
        K = np.concatenate( (K, np.array([[0,0,0,1]])), axis = 0 )
        return np.float32(K)

    def get_sensor_map(self, cam_token, ratio, delta_u, delta_v, do_flip,
            sensor_type='radar'):

        point_cloud_uv = self.nusc_proc.get_proj_dist_sensor(
                cam_token, sensor_type=sensor_type)

        point_cloud_uv = self.nusc_proc.adjust_cloud_uv(point_cloud_uv,
                self.width, self.height, ratio, delta_u, delta_v)

        depth_map = self.nusc_proc.make_depthmap(
                point_cloud_uv, (self.height, self.width))

        if do_flip:
            depth_map = np.flip(depth_map, axis = 1)

        return depth_map
