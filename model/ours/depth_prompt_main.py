import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import BaseModule
from .mde_ops import resize
from .mde_encoder_swin import DepthFormerSwin
from .mde_decoder import DenseDepthHead
from .mde_neck_hahi import HAHIHeteroNeck
from .common import *

class CSPNAccelerate(nn.Module):
    def __init__(self, kernel_size):
        super(CSPNAccelerate, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, kernel, input, input0):  
        bs = input.size()[0]
        h, w = input.size()[2], input.size()[3]

        input_im2col = F.unfold(input, self.kernel_size, 1, self.kernel_size//2, 1)

        kernel = kernel.reshape(bs, self.kernel_size * self.kernel_size, h * w)

        input0 = input0.view(bs, 1, h * w)
        mid_index = int((self.kernel_size * self.kernel_size - 1) / 2)
        input_im2col[:, mid_index:mid_index + 1, :] = input0

        output = (input_im2col * kernel).sum(dim=1)
        return output.view(bs, 1, h, w)

class depthprompting(nn.Module):
    def __init__(self, args):
        super(depthprompting, self).__init__()

        self.args = args
        self.prop_layer = CSPNAccelerate(args.prop_kernel)
        self.num_neighbors = self.args.prop_kernel * self.args.prop_kernel

        self.conv1_dep = conv_bn_relu(1, 64, kernel=3, stride=1, padding=1,
                                      bn=False)
        net = get_resnet34(not self.args.no_res_pre)

        self.conv2 = net.layer1
        self.conv3 = net.layer2
        self.conv4 = net.layer3
        self.conv5 = net.layer4

        del net

        self.conv6 = conv_bn_relu(512, 512, kernel=3, stride=2, padding=1)
        self.conv7 = conv_bn_relu(512, 512, kernel=3, stride=2, padding=1)


        self.dec6 = convt_bn_relu(1536+512, 512, kernel=3, stride=2,
                                  padding=1, output_padding=1)
        self.dec5 = convt_bn_relu(768+512+512, 512, kernel=3, stride=2,
                                  padding=1, output_padding=1)
        self.dec4 = convt_bn_relu(384+512+512, 512, kernel=3, stride=2,
                                  padding=1, output_padding=1)
        self.dec3 = convt_bn_relu(192+512+256, 256, kernel=3, stride=2,
                                  padding=1, output_padding=1)

        self.dec2 = convt_bn_relu(64+256+128, 128, kernel=3, stride=2,
                                  padding=1, output_padding=1)


        self.gd_dec1 = conv_bn_relu(128 + 64, 64, kernel=3, stride=1,
                                    padding=1)
        self.gd_dec0 = conv_bn_relu(64 + 64, self.num_neighbors, kernel=3, stride=1,
                                    padding=1, bn=False, relu=False)

        if self.args.conf_prop:
            self.cf_dec1 = conv_bn_relu(128+64, 32, kernel=3, stride=1,
                                        padding=1)
            self.cf_dec0 = nn.Sequential(
                nn.Conv2d(32+64, 1, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid()
            )

        params = []
        for param in self.named_parameters():
            if param[1].requires_grad:
                params.append(param[1])


        self.backbone = DepthFormerSwin()
        self.neck = HAHIHeteroNeck()
        self.decode_head = DenseDepthHead(max_depth = args.max_depth)
        self.align_corners = True 

        print("Monodcular Depth Model FREEZE !!")
        print("Bias Tuning !")
        for name, var in self.backbone.named_parameters():
            if not 'bias' in name:
                var.requires_grad = False
        for name, var in self.neck.named_parameters():
            if not 'bias' in name:
                var.requires_grad = False
        for name, var in self.decode_head.named_parameters():
            if not 'bias' in name:
                var.requires_grad = False

    def _concat(self, fd, fe, dim=1):
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        if Hd > He:
            h = Hd - He
            fd = fd[:, :, :-h, :]

        if Wd > We:
            w = Wd - We
            fd = fd[:, :, :, :-w]

        f = torch.cat((fd, fe), dim=dim)

        return f

    def forward(self, sample):
        
        x = self.backbone(sample['rgb'])
        
        x = self.neck(x)
        x_freeze_multiscale_image_feature = x
        x = self.decode_head(x) 
        x = torch.clamp(x, min=self.decode_head.min_depth, max=self.decode_head.max_depth)
        out = resize(input=x, size=sample['dep'].shape[2:], mode='bilinear', align_corners=self.align_corners)
        
        if sample['dep'].sum()==0.: 
            output = {'pred': out, 'pred_init': out, 'pred_inter': None, 'guidance': None, 'confidence': None}
            return output
        
        
        if self.args.init_scailing: 
            pred_init = out.detach().clone()
            for i in range(pred_init.shape[0]):
                dep_ = sample['dep'][i]
                idx_nnz = torch.nonzero(dep_.view(-1) > 0.0001, as_tuple=False)
                B = dep_.view(-1)[idx_nnz]
                A = pred_init[i].view(-1)[idx_nnz]
                num_dep = A.shape[0]
                if num_dep < 16: 
                    continue
                A = torch.cat((A,torch.ones(num_dep,1).to(A)),dim=1)
                X = torch.linalg.lstsq(A, B).solution
                X = X.to(pred_init)
                pred_init[i]  = pred_init[i] * X[0] + X[1]
        else:
            pred_init = out

        dep = sample['dep']
        fe1 = self.conv1_dep(dep)

        fe2 = self.conv2(fe1) 
        fe3 = self.conv3(fe2) 
        fe4 = self.conv4(fe3) 
        fe5 = self.conv5(fe4) 
        fe6 = self.conv6(fe5) 
        fe7 = self.conv7(fe6) 

        fd6 = self.dec6(self._concat(x_freeze_multiscale_image_feature[4], fe7)) 
        fd5 = self.dec5(self._concat(x_freeze_multiscale_image_feature[3],self._concat(fd6, fe6))) 
        fd4 = self.dec4(self._concat(x_freeze_multiscale_image_feature[2],self._concat(fd5, fe5))) 
        fd3 = self.dec3(self._concat(x_freeze_multiscale_image_feature[1],self._concat(fd4, fe4))) 
        fd2 = self.dec2(self._concat(x_freeze_multiscale_image_feature[0],self._concat(fd3, fe3))) 

        gd_fd1 = self.gd_dec1(self._concat(fd2, fe2)) 
        guide = self.gd_dec0(self._concat(gd_fd1, fe1)) 

        if self.args.conf_prop:
            cf_fd1 = self.cf_dec1(self._concat(fd2, fe2))
            confidence = self.cf_dec0(self._concat(cf_fd1, fe1))
        else:
            confidence = None

        depth = pred_init
        sparse_dep = dep
        sparse_mask = sparse_dep.sign()
        pred_inter = [pred_init]

        if self.args.conf_prop:
            sparse_dep = sparse_dep * confidence
        guide_sum = torch.sum(guide.abs(), dim=1, keepdim=True)
        guide = torch.div(guide, guide_sum)

        if self.args.data_name == 'NYU' or self.args.data_name == 'IPAD':
            for i in range(self.args.prop_time):
                depth = self.prop_layer(guide, depth, pred_init)
                depth = sparse_dep * sparse_mask + (1 - sparse_mask) * depth
                pred_inter.append(depth)

        elif self.args.data_name == 'KITTIDC':
            for i in range(self.args.prop_time):
                depth = sparse_dep * sparse_mask + (1 - sparse_mask) * depth
                depth = self.prop_layer(guide, depth, pred_init)
                pred_inter.append(depth)
        depth = torch.clamp(depth, min=0)

        output = {'pred': depth, 'pred_init': out, 'pred_inter': pred_inter, 'guidance': guide, 'confidence': confidence}

        return output

