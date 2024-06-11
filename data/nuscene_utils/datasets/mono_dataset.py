from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import cv2
import copy
from PIL import Image 

import torch
import torch.utils.data as data
from torchvision import transforms

from lib.utils import image_resize

def pil_loader(path):

    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class MonoDataset(data.Dataset):

    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg', 
                 not_do_color_aug=False,
                 not_do_flip=False,
                 do_crop=False,
                 crop_bound=[0.0, 1.0], #
                 seg_mask='none',
                 boxify=False,
                 MIN_OBJECT_AREA=20,
                 use_radar=False,
                 use_lidar=False,
                 prob_to_mask_objects=0.0, **kwargs):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext
        self.do_crop = do_crop
        self.crop_bound = crop_bound
        self.not_do_color_aug = not_do_color_aug
        self.not_do_flip = not_do_flip
        self.seg_mask = seg_mask
        self.boxify = boxify
        self.MIN_OBJECT_AREA = MIN_OBJECT_AREA
        self.use_radar = use_radar
        self.use_lidar = use_lidar
        self.prob_to_mask_objects = prob_to_mask_objects

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}

        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):

        for k in list(inputs):
            if "color" in k or "mask" in k:
                n, im, _ = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
            elif "mask" in k:
                n, im, i = k
                if self.seg_mask != 'none':
                    inputs[(n, im, i)] = torch.from_numpy(np.array(f))
                else:
                    inputs[(n, im, i)] = self.to_tensor(f)
            elif "radar" in k or "lidar" in k:
                n, im, i = k
                inputs[(n, im, i)] = torch.from_numpy(np.array(f))

        if self.seg_mask != 'none':
            self.process_masks(inputs, self.seg_mask)

            if random.random() < self.prob_to_mask_objects:
                self.mask_objects(inputs)

    def process_masks(self, inputs, mask_mode):
        """
        """
        MIN_OBJECT_AREA = self.MIN_OBJECT_AREA

        for scale in range(self.num_scales):

            if mask_mode == 'color':
                object_ids = torch.unique(torch.cat(
                    [inputs['mask', fid, scale] for fid in self.frame_idxs]),
                    sorted=True)
            else:
                object_ids = torch.Tensor([0, 255])

            for fid in self.frame_idxs:
                current_mask = inputs['mask', fid, scale]

                def process_obj_mask(obj_id, mask_mode=mask_mode):
                    if mask_mode == 'color':
                        mask = torch.logical_and(
                                torch.eq(current_mask, obj_id),
                                torch.ne(current_mask, 0)
                                )
                    else:
                        mask = torch.ne(current_mask, 0)

                    obj_size = torch.sum(mask)
                    if MIN_OBJECT_AREA != 0:
                        mask = torch.logical_and(mask, obj_size > MIN_OBJECT_AREA)
                    if not self.boxify:
                      return mask
                    binary_obj_masks_y = torch.any(mask, axis=1, keepdim=True)
                    binary_obj_masks_x = torch.any(mask, axis=0, keepdim=True)
                    return torch.logical_and(binary_obj_masks_y, binary_obj_masks_x)

                object_mask = torch.stack(
                        list(map(process_obj_mask, object_ids))
                        )
                object_mask = torch.any(object_mask, axis=0, keepdim=True)
                inputs['mask', fid, scale] = object_mask.to(torch.float32)

    def get_image(self, image, do_flip, crop_offset=-3, inter=cv2.INTER_AREA):
        if crop_offset == -3:            
            image, ratio, delta_u, delta_v = image_resize(
                    image, self.height, self.width, 0.0, 0.0, inter=inter) 

        else:
            raw_w, raw_h = image.size
            resize_w = self.width
            resize_h = int(raw_h * resize_w / raw_w)
            image, ratio, delta_u, delta_v = image_resize(
                    image, resize_h, resize_w, 0.0, 0.0, inter=inter)
            top = int(self.crop_bound[0] * resize_h)

            if crop_offset == -1:
                assert bottom >= top, "Not enough height to crop, please set a larger crop_bound range"
                crop_offset = np.random.randint(top, bottom + 1)
            elif crop_offset == -2:
                crop_offset = int((top+bottom)/2)

            image = np.array(image)
            image = image[crop_offset: crop_offset + self.height]
            image = Image.fromarray(image)
            delta_v += crop_offset

        if do_flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        return image, ratio, delta_u, delta_v, crop_offset

    def adjust_intrinsics(self, cam_intrinsics_mat, inputs, ratio, delta_u, delta_v, do_flip):

        for scale in range(self.num_scales):
            
            K = cam_intrinsics_mat.copy()

            K[0, :] *= ratio
            K[1, :] *= ratio
            K[0,2] -= delta_u
            K[1,2] -= delta_v

            if do_flip:
                K[0,2] = self.width - K[0,2]
            
            K[0, :] /= (2 ** scale)
            K[1, :] /= (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

    def mask_objects(self, inputs):
        for scale in range(self.num_scales):
            for fid in self.frame_idxs:
                inputs['color_aug', fid, scale] *= (1 - inputs['mask', fid, scale])
                inputs['color', fid, scale] *= (1 - inputs['mask', fid, scale])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_color_aug = (not self.not_do_color_aug) and do_color_aug
        do_flip = self.is_train and random.random() > 0.5
        do_flip = (not self.not_do_flip) and do_flip

        line = self.filenames[index].split()

        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:
            if i == "s": 
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(
                        folder, frame_index, other_side, do_flip
                        )
            else:
                inputs[("color", i, -1)] = self.get_color(
                        folder, frame_index + i, side, do_flip
                        )

        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def get_mask(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
