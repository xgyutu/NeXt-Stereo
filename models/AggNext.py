import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

# from mednextv1.blocks import *
import math
from models.submodule import *


class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class channelAtt(SubModule):
    def __init__(self, cv_chan, im_chan):
        super(channelAtt, self).__init__()

        self.im_att = nn.Sequential(
            BasicConv(im_chan, im_chan // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(im_chan // 2, cv_chan, 1))

        self.weight_init()

    def forward(self, cv, im):
        channel_att = self.im_att(im).unsqueeze(2)
        cv = torch.sigmoid(channel_att) * cv
        return cv


class Block(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 exp_r: int = 4,
                 kernel_size: int = 7,
                 do_res: int = True,
                 n_groups: int or None = None,

                 ):
        super().__init__()

        self.do_res = do_res

        # First convolution layer with DepthWise Convolutions

        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels if n_groups is None else n_groups,
        )

        # Normalization Layer. GroupNorm is used by default.
        self.norm = nn.GroupNorm(
            num_groups=in_channels,
            num_channels=in_channels
        )

        # Second convolution (Expansion) layer with Conv3D 1x1x1
        self.conv2 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # GeLU activations
        self.act = nn.GELU()

        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = nn.Conv3d(
            in_channels=exp_r * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x, dummy_tensor=None):
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))
        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1
        return x1


class DownBlock(Block):

    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7,
                 do_res=False):

        super().__init__(in_channels, out_channels, exp_r, kernel_size,
                         do_res=False)

        self.resample_do_res = do_res
        if do_res:
            self.res_conv = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2
            )

        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x, dummy_tensor=None):

        x1 = super().forward(x)

        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res

        return x1


class UpBlock(Block):

    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7,
                 do_res=False):
        super().__init__(in_channels, out_channels, exp_r, kernel_size,
                         do_res=False)

        self.resample_do_res = do_res
        if do_res:
            self.res_conv = nn.ConvTranspose3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2
            )

        self.conv1 = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x, dummy_tensor=None):

        x1 = super().forward(x)
        # Asymmetry but necessary to match shape
        x1 = torch.nn.functional.pad(x1, (1, 0, 1, 0, 1, 0))

        if self.resample_do_res:
            res = self.res_conv(x)
            res = torch.nn.functional.pad(res, (1, 0, 1, 0, 1, 0))
            x1 = x1 + res

        return x1


class AggNeXt(nn.Module):

    def __init__(self,
                 in_channels=8,
                 n_channels=8,
                 n_classes=1,
                 exp_r=3,  # Expansion ratio as in Swin Transformers
                 kernel_size=5,  # Ofcourse can test kernel_size
                 do_res=True,  # Can be used to individually test residual connection
                 do_res_up_down=True,  # Additional 'res' connection on up and down convs
                 block_counts: list = [2, 2, 1, 1, 1, 1, 1, 2, 2],  # Can be used to test staging ratio:
                 # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
                 CGE=True,
                 ):
        super().__init__()

        self.inside_block_checkpointing = False
        self.outside_block_checkpointing = False
        self.CGE = CGE

        if type(exp_r) == int:
            exp_r = [exp_r for i in range(len(block_counts))]

        self.enc_block_0 = nn.Sequential(*[
            Block(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[0],
                kernel_size=kernel_size,
                do_res=do_res,
            )
            for i in range(block_counts[0])]
                                         )

        self.down_0 = DownBlock(
            in_channels=n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[1],
            kernel_size=kernel_size,
            do_res=do_res_up_down,
        )

        self.enc_block_1 = nn.Sequential(*[
            Block(
                in_channels=n_channels * 2,
                out_channels=n_channels * 2,
                exp_r=exp_r[1],
                kernel_size=kernel_size,
                do_res=do_res,
            )
            for i in range(block_counts[1])]
                                         )

        self.down_1 = DownBlock(
            in_channels=2 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[2],
            kernel_size=kernel_size,
            do_res=do_res_up_down,
        )

        self.enc_block_2 = nn.Sequential(*[
            Block(
                in_channels=n_channels * 4,
                out_channels=n_channels * 4,
                exp_r=exp_r[2],
                kernel_size=kernel_size,
                do_res=do_res,
            )
            for i in range(block_counts[2])]
                                         )

        self.down_2 = DownBlock(
            in_channels=4 * n_channels,
            out_channels=6 * n_channels,
            exp_r=exp_r[3],
            kernel_size=kernel_size,
            do_res=do_res_up_down,
        )

        self.enc_block_3 = nn.Sequential(*[
            Block(
                in_channels=n_channels * 6,
                out_channels=n_channels * 6,
                exp_r=exp_r[3],
                kernel_size=kernel_size,
                do_res=do_res,
            )
            for i in range(block_counts[3])]
                                         )

        self.down_3 = DownBlock(
            in_channels=6 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[4],
            kernel_size=kernel_size,
            do_res=do_res_up_down,
        )

        self.bottleneck = nn.Sequential(*[
            Block(
                in_channels=n_channels * 8,
                out_channels=n_channels * 8,
                exp_r=exp_r[4],
                kernel_size=kernel_size,
                do_res=do_res,
            )
            for i in range(block_counts[4])]
                                        )

        self.up_3 = UpBlock(
            in_channels=8 * n_channels,
            out_channels=6 * n_channels,
            exp_r=exp_r[5],
            kernel_size=kernel_size,
            do_res=do_res_up_down,
        )

        self.dec_block_3 = nn.Sequential(*[
            Block(
                in_channels=n_channels * 6,
                out_channels=n_channels * 6,
                exp_r=exp_r[5],
                kernel_size=kernel_size,
                do_res=do_res,
            )
            for i in range(block_counts[5])]
                                         )

        self.up_2 = UpBlock(
            in_channels=6 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[6],
            kernel_size=kernel_size,
            do_res=do_res_up_down,
        )

        self.dec_block_2 = nn.Sequential(*[
            Block(
                in_channels=n_channels * 4,
                out_channels=n_channels * 4,
                exp_r=exp_r[6],
                kernel_size=kernel_size,
                do_res=do_res,
            )
            for i in range(block_counts[6])]
                                         )

        self.up_1 = UpBlock(
            in_channels=4 * n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[7],
            kernel_size=kernel_size,
            do_res=do_res_up_down,
        )

        self.dec_block_1 = nn.Sequential(*[
            Block(
                in_channels=n_channels * 2,
                out_channels=n_channels * 2,
                exp_r=exp_r[7],
                kernel_size=kernel_size,
                do_res=do_res,
            )
            for i in range(block_counts[7])]
                                         )

        self.up_0 = UpBlock(
            in_channels=2 * n_channels,
            out_channels=n_channels,
            exp_r=exp_r[8],
            kernel_size=kernel_size,
            do_res=do_res_up_down,
        )

        self.dec_block_0 = nn.Sequential(*[
            Block(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[8],
                kernel_size=kernel_size,
                do_res=do_res,
            )
            for i in range(block_counts[8])]
                                         )

        self.out_0 = nn.Conv3d(n_channels, n_classes, kernel_size=1)

        # Used to fix PyTorch checkpointing bug
        self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)

        self.block_counts = block_counts

        # if self.CGE:
        #     self.feature_att_4 = channelAtt(in_channels * 1, 96)
        #     self.feature_att_8 = channelAtt(in_channels * 2, 64)
        #     self.feature_att_16 = channelAtt(in_channels * 4, 192)
        #     self.feature_att_32 = channelAtt(in_channels * 6, 160)
        #     self.feature_att_up_32 = channelAtt(in_channels * 6, 160)
        #     self.feature_att_up_16 = channelAtt(in_channels * 4, 192)
        #     self.feature_att_up_8 = channelAtt(in_channels * 2, 64)
        #     self.feature_att_up_4 = channelAtt(in_channels * 1, 96)

    def forward(self, x, imgs):
        x_res_0 = self.enc_block_0(x)
        # x_res_0 = self.feature_att_4(x_res_0, imgs[0])

        x = self.down_0(x_res_0)
        x_res_1 = self.enc_block_1(x)
        # x_res_1 = self.feature_att_8(x_res_1, imgs[1])
        x = self.down_1(x_res_1)
        x_res_2 = self.enc_block_2(x)
        # x_res_2 = self.feature_att_16(x_res_2, imgs[2])
        x = self.down_2(x_res_2)
        x_res_3 = self.enc_block_3(x)
        # x_res_3 = self.feature_att_32(x_res_3, imgs[3])
        x = self.down_3(x_res_3)

        x = self.bottleneck(x)

        x_up_3 = self.up_3(x)
        dec_x = x_res_3 + x_up_3
        x = self.dec_block_3(dec_x)
        # x = self.feature_att_up_32(x, imgs[3])

        del x_res_3, x_up_3

        x_up_2 = self.up_2(x)
        dec_x = x_res_2 + x_up_2
        x = self.dec_block_2(dec_x)
        # x = self.feature_att_up_16(x, imgs[2])

        del x_res_2, x_up_2

        x_up_1 = self.up_1(x)
        dec_x = x_res_1 + x_up_1
        x = self.dec_block_1(dec_x)
        # x = self.feature_att_up_8(x, imgs[1])

        del x_res_1, x_up_1

        x_up_0 = self.up_0(x)
        dec_x = x_res_0 + x_up_0
        x = self.dec_block_0(dec_x)
        # x = self.feature_att_up_4(x, imgs[0])
        del x_res_0, x_up_0, dec_x

        x = self.out_0(x)

        return x

