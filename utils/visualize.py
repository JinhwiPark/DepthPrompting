import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image
import os
import cv2
import warnings
warnings.filterwarnings('ignore')

def dilation(image):
    kernel = np.ones((3, 3), np.uint8) * 1
    return cv2.dilate(image, kernel, iterations=2) 

class visualize():
    def __init__(self, args, color_mode='jet'):
        self.cm = plt.get_cmap(color_mode)
        self.dilation_size = args.dilation_size
        self.max_depth = args.max_depth
        self.args = args

        self.img_mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        self.img_std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

    def data_put(self, sample, output):
        self.rgb = sample['rgb']
        self.dep = sample['dep'].detach()
        try:
            self.initial_pred = output['pred_init'].detach()
        except:
            self.initial_pred = output['pred'].detach()
        self.pred = output['pred'].detach()
        self.gt = sample['gt'].detach()
        try:
            self.guidance = output['guidance'].detach()
        except:
            self.guidance = output['pred'].detach()
        

    def depth(self, type, idx, path_to_save):
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        
        if type == 'sparse':
            x = self.dep
        elif type == 'pred':
            x = self.pred
        elif type == 'gt':
            x = self.gt
        elif type == 'initial':
            x = self.initial_pred
        elif type == 'all':
            pass
        else:
            raise Exception('Choose from sparse, pred, gt and all')

        if x == None:
            return

        x = x.data.cpu()
        if self.args.data_name == "IPAD":
            x = x[0][0].numpy() / x.max().item()

        x_2 = x[0][0].numpy() / self.max_depth
        x_2 = (255.0 * self.cm(x_2)).astype('uint8')

        if type == 'sparse':
            x_2 = dilation(x_2)
        x_2 = Image.fromarray(x_2[:, :, :3], 'RGB')

        path_save = '{}/{}_{}_2.png'.format(path_to_save, idx, type)
        x_2.save(path_save)

    def depth_iteration(self, idx, path_to_save):
        depth_list = self.depth_list
        if depth_list == None:
            return
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        for i, x in enumerate(depth_list):
            x = x.data.cpu()
            x = x[0][0].numpy() / self.max_depth
            x = (255.0 * self.cm(x)).astype('uint8')
            x = Image.fromarray(x[:, :, :3], 'RGB')

            path_save = '{}/{}_{}.png'.format(path_to_save, idx, i)
            x.save(path_save)

    def error_map(self, type, idx, path_to_save):
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        gt = self.gt
        gt = gt.data.cpu()
        gt = gt[0][0]

        pred = self.pred
        pred = pred.data.cpu()
        pred = pred[0][0]

        if type == 'l1':
            error = (gt - pred).abs()
        elif type == 'l2':
            error = (gt - pred) ** 2
        else:
            Exception('Choose from l1, l2')

        error = error.numpy()
        plt.figure(figsize=(8, 6))
        plt.imshow(error)
        plt.colorbar()
        plt.show()
        plt.savefig('{}/{}_{}.png'.format(path_to_save, idx, type))

    def RGB(self, idx, path_to_save):

        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        x = self.rgb
        x.mul_(self.img_std.type_as(x)).add_(self.img_mean.type_as(x))
        x = x.data.cpu()
        x = np.transpose(x[0].numpy(), (1, 2, 0))
        x = (255.0 * x)
        x = np.clip(x, 0, 256).astype('uint8')
        x = Image.fromarray(x, 'RGB')
        path_save = '{}/{}_rgb.png'.format(path_to_save, idx)
        x.save(path_save)

    def save_all_nyu_gt_sparse_rgb_errormap(self, idx, path_to_save):
        self.path_output = path_to_save
        os.makedirs(self.path_output, exist_ok=True)

        print('start to save image-idx[%d]' % (idx))

        pred = torch.clamp(self.pred, min=0)
        rgb = self.rgb
        dep = self.dep
        gt = self.gt

        rgb.mul_(self.img_std.type_as(rgb)).add_(self.img_mean.type_as(rgb))

        rgb = rgb[0, :, :, :].data.cpu().numpy()
        dep = dep[0, 0, :, :].data.cpu().numpy()
        pred = pred[0, 0, :, :].data.cpu().numpy()
        gt = gt[0, 0, :, :].data.cpu().numpy()

        dep_mask = np.where([dep>0])
        dep_value = dep[np.where(dep>0)]
        dep_x, dep_y = dep_mask[1], dep_mask[2]

        rgb = 255.0 * np.transpose(rgb, (1, 2, 0))
        dep = dep / self.args.max_depth
        pred = pred / self.args.max_depth
        gt = gt / self.args.max_depth 

        mindepth = min(gt.min(), pred.min()) 
        maxdepth = max(gt.max(), pred.max()) 
        dpi_ = 500
        num_col = 3
        num_row = 2
        idx_fig = 0
        plt.figure(figsize=(num_col * 8, num_row * 6))
        

        idx_fig = idx_fig + 1
        plt.subplot(num_row, num_col, idx_fig)
        plt.imshow(np.uint8(rgb), aspect='auto')
        plt.scatter(dep_y, dep_x, c=dep_value, s=20, cmap=self.cm,marker='s')
        plt.axis('off')
        plt.tight_layout()
        plt.colorbar()
        plt.title('rgb')

        idx_fig = idx_fig + 1
        plt.subplot(num_row, num_col, idx_fig)
        plt.imshow(dilation(dep), aspect='auto', cmap=self.cm, interpolation='nearest')
        plt.axis('off')
        plt.tight_layout()
        plt.colorbar()
        plt.title('input sparser depth')

        idx_fig = idx_fig + 1
        plt.subplot(num_row, num_col, idx_fig)
        plt.imshow(pred, aspect='auto', vmin=mindepth, vmax=maxdepth, cmap=self.cm, interpolation='nearest')
        plt.axis('off')
        plt.tight_layout()
        plt.colorbar()
        plt.title('pred depth')

        idx_fig = idx_fig + 1
        plt.subplot(num_row, num_col, idx_fig)
        plt.imshow(dilation(gt), aspect='auto', vmin=mindepth, vmax=maxdepth, cmap=self.cm, interpolation='nearest')
        plt.axis('off')
        plt.tight_layout()
        plt.colorbar()
        plt.title('gt depth')

        idx_fig = idx_fig + 1
        plt.subplot(num_row, num_col, idx_fig)
        err = np.abs(gt - pred)
        plt.imshow(err, aspect='auto', cmap='jet', interpolation='nearest', vmin=0, vmax=0.01)
        plt.axis('off')
        plt.tight_layout()
        plt.colorbar()
        plt.title('error map_1')

        idx_fig = idx_fig + 1
        plt.subplot(num_row, num_col, idx_fig)
        err = np.abs(gt - pred)
        plt.imshow(err, aspect='auto', cmap='jet', interpolation='nearest', vmin=0, vmax=0.20)
        plt.axis('off')
        plt.tight_layout()
        plt.colorbar()
        plt.title('error map_4')

        plt.savefig(os.path.join(self.path_output, '{:04d}.svg'.format(idx)))
        plt.close('all')


    def save_all_kitti_gt_sparse_rgb_errormap(self, idx, path_to_save):
        self.path_output = path_to_save
        os.makedirs(self.path_output, exist_ok=True)

        print('start to save image-idx[%d]' % (idx))

        pred = torch.clamp(self.pred, min=0)
        init_depth = torch.clamp(self.initial_pred, min=0)
        rgb = self.rgb
        dep = self.dep
        gt = self.gt

        rgb.mul_(self.img_std.type_as(rgb)).add_(self.img_mean.type_as(rgb))

        rgb = rgb[0, :, :, :].data.cpu().numpy()
        dep = dep[0, 0, :, :].data.cpu().numpy()
        pred = pred[0, 0, :, :].data.cpu().numpy()
        init_depth = init_depth[0, 0, :, :].data.cpu().numpy()
        gt = gt[0, 0, :, :].data.cpu().numpy()

        rgb = 255.0 * np.transpose(rgb, (1, 2, 0))
        dep = dep / self.args.max_depth
        pred = pred / self.args.max_depth
        init_depth = init_depth / self.args.max_depth
        gt = gt / self.args.max_depth

        mindepth = max(gt.min(), pred.min())
        maxdepth = max(gt.max(), pred.max())

        dpi_ = 500
        num_col = 1
        num_row = 5
        idx_fig = 0
        plt.figure(figsize=(num_col * 36, num_row * 6))
        
        colormap = cm.get_cmap(self.args.color_mode, 256)
        newcmp = ListedColormap(colormap(np.linspace(0.09, 1, 256)))
        self.cm = newcmp
        
        idx_fig = idx_fig + 1
        plt.subplot(num_row, num_col, idx_fig)
        plt.imshow(np.uint8(rgb), aspect='auto')
        plt.axis('off')
        plt.tight_layout()
        plt.colorbar()
        plt.title('rgb')

        idx_fig = idx_fig + 1
        plt.subplot(num_row, num_col, idx_fig)
        plt.imshow(dilation(dep), aspect='auto', vmin=mindepth, vmax=maxdepth, cmap=self.cm, interpolation='nearest')
        plt.axis('off')
        plt.tight_layout()
        plt.colorbar()
        plt.title('input sparser depth')

        idx_fig = idx_fig + 1
        plt.subplot(num_row, num_col, idx_fig)
        plt.imshow(pred, aspect='auto', vmin=mindepth, vmax=maxdepth, cmap=self.cm, interpolation='nearest')
        plt.axis('off')
        plt.tight_layout()
        plt.colorbar()
        plt.title('pred depth')

        idx_fig = idx_fig + 1
        plt.subplot(num_row, num_col, idx_fig)
        plt.imshow(dilation(gt), aspect='auto', vmin=mindepth, vmax=maxdepth, cmap=self.cm, interpolation='nearest')
        plt.axis('off')
        plt.tight_layout()
        plt.colorbar()
        plt.title('gt depth')

        idx_fig = idx_fig + 1
        plt.subplot(num_row, num_col, idx_fig)
        err = np.abs(gt - pred)
        plt.imshow(err, aspect='auto', cmap='jet', interpolation='nearest', vmin=0, vmax=0.01)
        plt.axis('off')
        plt.tight_layout()
        plt.colorbar()
        plt.title('error map_1')

        plt.savefig(os.path.join(self.path_output, '{:04d}.svg'.format(idx)))
        plt.close('all')

