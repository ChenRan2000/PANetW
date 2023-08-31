#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

# import torch
# import torch.nn as nn

# from .darknet import CSPDarknet
# from .network_blocks import BaseConv, CSPLayer, DWConv, get_activation, Bottleneck



# class YOLOPAFPN(nn.Module):
#     """
#     YOLOv3 model. Darknet 53 is the default backbone of this model.
#     """

#     def __init__(
#         self,
#         depth=1.0,
#         width=1.0,
#         # in_features = ("dark5"),
#         in_features=("dark3", "dark4", "dark5"),
#         in_channels=[256, 512, 1024],
#         depthwise=False,
#         act="silu",
#     ):
#         self.action="silu"
#         super().__init__()
#         self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
#         self.in_features = in_features
#         self.in_channels = in_channels
#         Conv = DWConv if depthwise else BaseConv
#         # self.down_conv = BaseConv(int(in_channels[1]*width), int(in_channels[0]*width), 1, 1, act=act)

#         def get_activation(name="silu", inplace=True):
#             if name == "silu"  :
#                 module = nn.SiLU(inplace=inplace)
#             elif name == "relu":
#                 module = nn.ReLU(inplace=inplace)
#             elif name == "lrelu":
#                 module = nn.LeakyReLU(0.1, inplace=inplace)
#             else:
#                 raise AttributeError("Unsupported act type: {}".format(name))
#             return module

#         class Bottle(nn.Module):
        
#             def __init__(self,
#                          in_channels: int = 512,
#                          mid_channels: int = 128,
#                          dilation: int = 1,
#                          norm_type: str = 'BN',
#                          act_type: str = 'silu'):
#                 super(Bottle, self).__init__()
#                 self.conv1 = nn.Sequential(
#                     nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
#                     nn.BatchNorm2d(mid_channels),
#                     get_activation(act_type)
#                 )
#                 self.conv2 = nn.Sequential(
#                     nn.Conv2d(mid_channels, mid_channels,
#                               kernel_size=3, padding=dilation, dilation=dilation),
#                     nn.BatchNorm2d(mid_channels),
#                     get_activation(act_type)
#                 )
#                 self.conv3 = nn.Sequential(
#                     nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0),
#                     nn.BatchNorm2d(in_channels),
#                     get_activation(act_type)
#                 )
        
#             def forward(self, x: torch.Tensor) -> torch.Tensor:
#                 identity = x
#                 out = self.conv1(x)
#                 out = self.conv2(out)
#                 out = self.conv3(out)
#                 out = out + identity
#                 return out


#         self.fpn_conv2 = BaseConv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 1, act=act)
#         self.fpn_conv1 = BaseConv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 1, act=act)
#         self.fpn_conv0 = BaseConv(int(in_channels[2] * width), int(in_channels[2] * width), 3, 1, act=act)

#         dilations = [1,2,3,4,5,6,7,8]
        
#         encoder_blocks0 = []
#         self.num_residual_blocks = 8
        
    
#         for i in range(self.num_residual_blocks):  # 8
#             encoder_blocks0.append(
#                 Bottle(
#                     int(in_channels[2] * width),
#                     int(in_channels[2] * width * 0.25),
#                     dilation=dilations[i],
#                     norm_type=nn.BatchNorm2d,
#                     act_type=self.action,
#                 )
#             )
#         self.dilated_encoder_blocks0 = nn.Sequential(*encoder_blocks0)

#         encoder_blocks1 = []
#         for i in range(self.num_residual_blocks):  # 8
#             encoder_blocks1.append(
#                 Bottle(
#                     int(in_channels[1] * width),
#                     int(in_channels[1] * width* 0.25),
#                     dilation=dilations[i],
#                     norm_type=nn.BatchNorm2d,
#                     act_type=self.action,
#                 )
#             )
#         self.dilated_encoder_blocks1 = nn.Sequential(*encoder_blocks1)

#         encoder_blocks2 = []
#         for i in range(self.num_residual_blocks):  # 8
#             encoder_blocks2.append(
#                 Bottle(
#                     int(in_channels[0] * width),
#                     int(in_channels[0] * width* 0.25),
#                     dilation=dilations[i],
#                     norm_type=nn.BatchNorm2d,
#                     act_type=self.action,
#                 )
#             )
#         self.dilated_encoder_blocks2 = nn.Sequential(*encoder_blocks2)

#         self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
#         self.lateral_conv0 = BaseConv(
#             int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
#         )
#         self.C3_p4 = CSPLayer(
#             int(2 * in_channels[1] * width),
#             int(in_channels[1] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )  # cat

#         self.reduce_conv1 = BaseConv(
#             int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
#         )
#         self.C3_p3 = CSPLayer(
#             int(2 * in_channels[0] * width),
#             int(in_channels[0] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         # bottom-up conv
#         self.bu_conv2 = Conv(
#             int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
#         )
#         self.C3_n3 = CSPLayer(
#             int(2 * in_channels[0] * width),
#             int(in_channels[1] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         # bottom-up conv
#         self.bu_conv1 = Conv(
#             int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
#         )
#         self.C3_n4 = CSPLayer(
#             int(2 * in_channels[1] * width),
#             int(in_channels[2] * width),
#             round(3 * depth),
#             False,
#             depthwise=depthwise,
#             act=act,
#         )

#         # self.out0 = Conv(int(in_channels[1] * width), int(in_channels[2] * width), 1, 1, act=act)
#         # self.out1 = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 1, 1, act=act)
#         # self.out2 = Conv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)


#     def forward(self, input):
#         """
#         Args:
#             inputs: input images.

#         Returns:
#             Tuple[Tensor]: FPN feature.
#         """

#         #  backbone
#         out_features = self.backbone(input)
#         features = [out_features[f] for f in self.in_features]
#         # features = [out_features[self.in_features]]
#         # [x0] = features
#         [x2, x1, x0] = features

#         fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
#         f_out0 = self.upsample(fpn_out0)  # 512/16
#         f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
#         f_out0 = self.C3_p4(f_out0)  # 1024->512/16

#         fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
#         f_out1 = self.upsample(fpn_out1)  # 256/8
#         f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
#         pan_out2 = self.C3_p3(f_out1)  # 512->256/8

#         p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
#         p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
#         pan_out1 = self.C3_n3(p_out1)  # 512->512/16

#         p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
#         p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
#         pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

#         # pan_out0 = self.lateral_conv0(x0)  # 1024->512/32
#         # pan_out0 = self.lateral_conv0(pan_out0)  # 1024->512/32
#         pan_out0 = self.fpn_conv0(pan_out0)  # 1024
#         pan_out0 = self.dilated_encoder_blocks0(pan_out0)
#         # pan_out0 = self.down_conv(pan_out0)
#         # pan_out0 = self.out0(pan_out0)

#         pan_out1 = self.fpn_conv1(pan_out1) # 512
#         pan_out1 = self.dilated_encoder_blocks1(pan_out1)
#         # pan_out1 = self.down_conv(pan_out1)
#         # pan_out1 = self.out1(pan_out1)

#         # pan_out2 = self.upsample(pan_out2) # 256 -> 512
#         pan_out2 = self.fpn_conv2(pan_out2) # 256
#         pan_out2 = self.dilated_encoder_blocks2(pan_out2)
#         # pan_out2 = self.down_conv(pan_out2)
#         # pan_out2 = self.out2(pan_out2)

#         out_puts = (pan_out2, pan_out1, pan_out0)
#         # out_puts = (pan_out0)
#         return out_puts

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs
