from inspect import CO_VARARGS
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch.nn.functional import embedding
from torch.nn.modules import conv

import torch.nn.functional as F
from .mde_model_utils import UpConvBlock, BasicConvBlock

class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super(UpSample, self).__init__()
        self.convA = ConvModule(skip_input, output_features, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.convB = ConvModule(output_features, output_features, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.convB(self.convA(torch.cat([up_x, concat_with], dim=1)))

class DenseDepthHead(BaseModule):
    """DenseDepthHead.
    This head is implemented of `DenseDepth: <https://arxiv.org/abs/1812.11941>`_.
    Args:
        up_sample_channels (List): Out channels of decoder layers.
        fpn (bool): Whether apply FPN head.
            Default: False
        conv_dim (int): Default channel of features in FPN head.
            Default: 256.
    """
    def __init__(self,
                 in_channels=[64, 192, 384, 768, 1536],
                 up_sample_channels=[64, 192, 384, 768, 1536],
                 fpn=False,
                 conv_dim=256,
                 norm_cfg=None,
                 act_cfg=dict(type='LeakyReLU', inplace=True),
                 min_depth= 0.001,
                 max_depth=80,
                 align_corners=False,
                 channels=64
                 ):
        super(DenseDepthHead, self).__init__()

        self.up_sample_channels = up_sample_channels[::-1]
        self.in_channels = in_channels[::-1]
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.conv_list = nn.ModuleList()
        up_channel_temp = 0

        self.fpn = fpn

        self.align_corners = align_corners
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.channels = channels
        self.relu = nn.ReLU()

        self.conv_depth = nn.Conv2d(channels, 1, kernel_size=3, padding=1, stride=1)


        if self.fpn:
            self.num_fpn_levels = len(self.in_channels)

            # construct the FPN
            self.lateral_convs = nn.ModuleList()
            self.output_convs = nn.ModuleList()

            for idx, in_channel in enumerate(self.in_channels[:self.num_fpn_levels]):
                lateral_conv = ConvModule(
                    in_channel, conv_dim, kernel_size=1, norm_cfg=self.norm_cfg
                )
                output_conv = ConvModule(
                    conv_dim,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                )
                self.lateral_convs.append(lateral_conv)
                self.output_convs.append(output_conv)

        else:
            for index, (in_channel, up_channel) in enumerate(
                    zip(self.in_channels, self.up_sample_channels)):
                if index == 0:
                    self.conv_list.append(
                        ConvModule(
                            in_channels=in_channel,
                            out_channels=up_channel,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            act_cfg=None
                        ))
                else:
                    self.conv_list.append(
                        UpSample(skip_input=in_channel + up_channel_temp,
                                 output_features=up_channel,
                                 norm_cfg=self.norm_cfg,
                                 act_cfg=self.act_cfg))
                # save earlier fusion target
                up_channel_temp = up_channel
                                    
    def depth_pred(self, feat):
        """Prediction each pixel."""
        output = self.relu(self.conv_depth(feat)) + self.min_depth
        return output

    def forward(self, inputs):
        """Forward function."""
        #### inputs ####
        #x[0] ([B, 64, H/2, W/2])                                          
        #x[1] ([B, 192, H/4, W/4])
        #x[2] ([B, 384, H/8, W/8])
        #x[3] ([B, 768, H/16, W/16])
        #x[4] ([B, 1536, H/32, W/32])
        
        # import pdb;pdb.set_trace()
        temp_feat_list = []
        if self.fpn:
            for index, feat in enumerate(inputs[::-1]):
                x = feat
                lateral_conv = self.lateral_convs[index]
                output_conv = self.output_convs[index]
                cur_fpn = lateral_conv(x)

                # Following FPN implementation, we use nearest upsampling here. Change align corners to True.
                if index != 0:
                    y = cur_fpn + F.interpolate(temp_feat_list[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=True)
                else:
                    y = cur_fpn
                    
                y = output_conv(y)
                temp_feat_list.append(y)

        else:
            temp_feat_list = []
            for index, feat in enumerate(inputs[::-1]):
                if index == 0:
                    temp_feat = self.conv_list[index](feat)
                    temp_feat_list.append(temp_feat)
                else:
                    skip_feat = feat
                    up_feat = temp_feat_list[index-1]
                    temp_feat = self.conv_list[index](up_feat, skip_feat)
                    temp_feat_list.append(temp_feat)

        output = self.depth_pred(temp_feat_list[-1])
        return output

