import torch
import torch.nn as nn
import math
import warnings

from timm.models.layers import DropPath
from torch.nn.modules.utils import _pair as to_2tuple
from torch.nn.modules.utils import _triple as to_3tuple
from mmseg.models.builder import BACKBONES
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
img_size_list = []
stride_list = []

class Mlp(BaseModule):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class StemConv(BaseModule):
    def __init__(self, in_channels, out_channels, norm_cfg=dict(type='GN', num_groups=4, requires_grad=True), strides=[(1, 2, 2), (1, 1, 1), (1, 2, 2)]):
        super(StemConv, self).__init__()

        self.proj_1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 2,
                      kernel_size=(3, 3, 3), stride=strides[0],  padding=(1, 1, 1)),
            build_norm_layer(norm_cfg, out_channels // 2)[1],

            nn.Conv3d(out_channels // 2, out_channels // 2,
                      kernel_size=(3, 3, 3), stride=strides[1], padding=(1, 1, 1)),
            build_norm_layer(norm_cfg, out_channels // 2)[1],

        )
        self.act = nn.GELU()
        self.proj_2 = nn.Sequential(
            nn.Conv3d(out_channels // 2, out_channels,
                      kernel_size=(3, 3, 3), stride=strides[2], padding=(1, 1, 1)),
            build_norm_layer(norm_cfg, out_channels)[1],
        )


    def forward(self, x):
        x_1 = self.act(self.proj_1(x))
        x = self.act(x_1)
        x = self.proj_2(x)
        _, _, D, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, x_1, D, H, W

class Attention2DModule(BaseModule):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv3d(dim, dim, (1, 3, 3), padding=(0, 1, 1), groups=dim)
        self.conv0_2 = nn.Conv3d(dim, dim, (1, 3, 3), padding=(0, 1, 1), groups=dim)

        self.conv1_1 = nn.Conv3d(dim, dim, (1, 1, 5), padding=(0, 0, 2), groups=dim)
        self.conv1_2 = nn.Conv3d(dim, dim, (1, 5, 1), padding=(0, 2, 0), groups=dim)

        self.conv2_1 = nn.Conv3d(
            dim, dim, (1, 1, 7), padding=(0, 0, 3), groups=dim)
        self.conv2_2 = nn.Conv3d(
            dim, dim, (1, 7, 1), padding=(0, 3, 0), groups=dim)
        self.conv3 = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u

class AttentionModule(BaseModule):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv3d(dim, dim, (3, 1, 5), padding=(1, 0, 2), groups=dim)
        self.conv0_2 = nn.Conv3d(dim, dim, (3, 5, 1), padding=(1, 2, 0), groups=dim)

        self.conv1_1 = nn.Conv3d(dim, dim, (3, 1, 7), padding=(1, 0, 3), groups=dim)
        self.conv1_2 = nn.Conv3d(dim, dim, (3, 7, 1), padding=(1, 3, 0), groups=dim)

        self.conv2_1 = nn.Conv3d(
            dim, dim, (3, 1, 11), padding=(1, 0, 5), groups=dim)
        self.conv2_2 = nn.Conv3d(
            dim, dim, (3, 11, 1), padding=(1, 5, 0), groups=dim)
        self.conv3 = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u



class SpatialAttention(BaseModule):
    def __init__(self, d_model, stage):
        super().__init__()

        global img_size_list
        img_size = img_size_list[stage]
        self.d_model = d_model
        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.spatial_gating_2d_unit = Attention2DModule(d_model)

        layer_scale_init_value = 1e-2
        self.layer_channel_scale = nn.Parameter(
            layer_scale_init_value * torch.ones((d_model, img_size[0])), requires_grad=True)
        self.layer_channel_attn = ChannelLayerAttention(d_model, img_size[0])
        self.layer_scale_3d = nn.Parameter(
            layer_scale_init_value * torch.ones((d_model)), requires_grad=True)


        self.proj_2 = nn.Conv3d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x_2d = self.spatial_gating_2d_unit(x)
        x = self.spatial_gating_unit(x) # MSCA
        x_2d = self.layer_channel_scale.unsqueeze(-1).unsqueeze(-1) * self.layer_channel_attn(x_2d)
        x = self.layer_scale_3d.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x
        x = x_2d + x
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(BaseModule):

    def __init__(self,
                 dim,
                 stage,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_cfg=dict(type='GN', num_groups=4, requires_grad=True),
                 ):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = SpatialAttention(dim, stage)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, D, H, W)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                               * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                               * self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x


class OverlapPatchEmbed(BaseModule):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=(2, 2, 2), in_chans=3, embed_dim=768, norm_cfg=dict(type='GN', num_groups=4, requires_grad=True)):
        super().__init__()
        patch_size = to_3tuple(patch_size)

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2))
        self.norm = build_norm_layer(norm_cfg, embed_dim)[1]

    def forward(self, x):
        x = self.proj(x)
        _, _, D, H, W = x.shape
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)

        return x, D, H, W

class MSCAN(BaseModule):
    def __init__(self,
                 in_chans=3,
                 out_chans=2,
                 embed_dims=[32, 64, 128, 256],
                 mlp_ratios=[4, 4, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[1, 1, 2, 1],
                 # depths=[3, 4, 6, 3],
                 num_stages=4,
                 norm_cfg=dict(type='GN', num_groups=4, requires_grad=True),
                 pretrained=None,
                 init_cfg=None,
                 img_size=[32,192,192]):
        super(MSCAN, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        cur = 0
        stride_list = [
            [(1, 1, 1), (1, 1, 1), (1, 2, 2)],    #第0层

            [(2, 2, 2)],  #第1层
            [(2, 2, 2)],  #第2层
            [(2, 2, 2)],  #第3层
        ]
        global img_size_list
        for stage in range(num_stages):
            for blk in stride_list[stage]:
                img_size[0] = int(img_size[0] / blk[0])
                img_size[1] = int(img_size[1] / blk[1])
                img_size[2] = int(img_size[2] / blk[2])
            img_size_list.append(img_size.copy())
        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(3, embed_dims[0], norm_cfg=norm_cfg, strides=stride_list[0])
            else:
                patch_embed = OverlapPatchEmbed(patch_size=7 if i == 0 else 3,
                                                stride=stride_list[i][0],
                                                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                                embed_dim=embed_dims[i],
                                                norm_cfg=norm_cfg)

            block = nn.ModuleList([Block(dim=embed_dims[i], stage=i, mlp_ratio=mlp_ratios[i],
                                         drop=drop_rate, drop_path=dpr[cur + j],
                                         norm_cfg=norm_cfg)
                                   for j in range(depths[i])])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        for i in reversed(range(num_stages)):
            if i == num_stages - 1:
                block = ConvU(chans=embed_dims[i],
                              scale_factor=(2, 2, 2),
                              first=True)
            elif i == 0:
                block = ConvU(chans=embed_dims[i],
                              scale_factor=(1, 2, 2),
                              first=False)
            else:
                block = ConvU(chans=embed_dims[i],
                              scale_factor=(2, 2, 2),
                              first=False)
            setattr(self, f"block_u{i + 1}", block)

        for i in reversed(range(3)):
            seg = nn.Conv3d(embed_dims[i], out_chans, 1)
            if i == 0:
                up = nn.Upsample(scale_factor=(1, 2, 2),
                    mode='trilinear', align_corners=False)
            else:
                up = nn.Upsample(scale_factor=(2, 2, 2),
                                 mode='trilinear', align_corners=False)
            setattr(self, f"seg{i + 1}", seg)
            setattr(self, f"up{i + 1}", up)




    def init_weights(self):
        print('init cfg', self.init_cfg)
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv3d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:

            super(MSCAN, self).init_weights()

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            if i == 0:
                x, x_1, D, H, W = patch_embed(x)
                outs.append(x_1)
            else:
                x, D, H, W = patch_embed(x)

            for blk in block:
                x = blk(x, D, H, W)
            x = norm(x)
            # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
            outs.append(x)

        up_res = []
        for i in reversed(range(self.num_stages)):
            block = getattr(self, f"block_u{i + 1}")
            x = block(x, outs[i])
            up_res.insert(0, x)

        res = None
        for i in reversed(range(3)):
            seg = getattr(self, f"seg{i + 1}")
            up = getattr(self, f"up{i + 1}")
            if i == 2:
                res = seg(up_res[i])
            else:
                res = seg(up_res[i]) + up(res)
        return res

class ConvU(nn.Module):
    def __init__(self,
                 chans=32,
                 scale_factor=(2, 2, 2),
                 norm_cfg=dict(type='GN', num_groups=4, requires_grad=True),
                 first=False):
        super(ConvU, self).__init__()
        self.scale_factor = scale_factor
        self.first = first

        if not self.first:
            self.conv1 = nn.Conv3d(2 * chans, chans, 3, 1, 1, bias=False)
            self.bn1   = build_norm_layer(norm_cfg, chans)[1]

        self.conv2 = nn.Conv3d(chans, chans // 2, 1, 1, 0, bias=False)
        self.bn2   = build_norm_layer(norm_cfg, chans // 2)[1]

        self.conv3 = nn.Conv3d(chans, chans, 3, 1, 1, bias=False)
        self.bn3   = build_norm_layer(norm_cfg, chans)[1]

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, prev):
        # final output is the localization layer
        if not self.first:
            x = self.relu(self.bn1(self.conv1(x)))

        y = F.upsample(x, scale_factor=self.scale_factor, mode='trilinear', align_corners=False)
        y = self.relu(self.bn2(self.conv2(y)))

        y = torch.cat([prev, y], 1)
        y = self.conv3(y)
        y = self.bn3(y)
        y = self.relu(y)

        return y
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class ChannelLayerAttention(nn.Module):
    def __init__(self, in_channels, in_depth, reduction_ratio=1):
        super(ChannelLayerAttention, self).__init__()

        # Calculate the number of reduced channels and depth
        reduced_channels = max(in_channels // reduction_ratio, 1)
        reduced_depth = max(in_depth // reduction_ratio, 1)

        self.avg_pool_c = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.avg_pool_d = nn.AdaptiveAvgPool3d((None, 1, 1))

        self.max_pool_c = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.max_pool_d = nn.AdaptiveMaxPool3d((None, 1, 1))

        self.fc_c = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False)
        )

        self.fc_d = nn.Sequential(
            nn.Linear(in_depth, reduced_depth, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_depth, in_depth, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, D, H, W = x.shape
        input = x
        # Channel Attention
        avg_out_c = self.fc_c(self.avg_pool_c(x).view(B, -1))
        max_out_c = self.fc_c(self.max_pool_c(x).view(B, -1))
        channel_attention = self.sigmoid(avg_out_c + max_out_c).view(B, -1, 1, 1, 1)
        x = x * channel_attention.expand_as(x) + x


        # Layer Attention
        avg_out_d = self.fc_d(self.avg_pool_d(x).view(B, C, D))
        max_out_d = self.fc_d(self.max_pool_d(x).view(B, C, D))
        layer_attention = self.sigmoid(avg_out_d + max_out_d)
        x = x * layer_attention.view(B, C, D, 1, 1) + x
        return x + input


if __name__ == '__main__':
    model = MSCAN()
    a = torch.rand((1,3, 32, 192, 192))
    b = model(a)
    #outs = (1, 16, 32, 256, 256)
    #outs = (1, 32, 32, 128, 128)
    #outs = (1, 64, 32, 64, 64)
    #outs = (1, 128, 32, 32, 32)
    #outs = (1, 256, 32, 16, 16)
    # model = ChannelLayerAttention(32, 16)
    # a = torch.rand((1, 32, 16, 64, 64))
    # b = model(a)
    print(b)
