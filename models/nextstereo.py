from __future__ import print_function
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.submodule import *
import math
import timm
from models.AggNext import AggNeXt
from models.CLKA_Refinement import CLKA_Refinement


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.LeakyReLU(0.1, True)  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU()

    def forward(self, x):
        # print(x)
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        # print(x)
        x = self.conv(x)
        return self.act(x)


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


class Feature(SubModule):
    def __init__(self):
        super(Feature, self).__init__()
        pretrained = True
        model = timm.create_model('mobilenetv2_100.ra_in1k', pretrained=pretrained, features_only=True)
        # print(model)
        layers = [1, 2, 3, 5, 6]
        chans = [16, 24, 32, 96, 160]
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        # self.act1 = model.act

        self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
        self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])

    def forward(self, x):
        x = self.bn1(self.conv_stem(x))
        x2 = self.block0(x)
        x4 = self.block1(x2)
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)
        # print("x4", x4.shape)
        # print("x8", x8.shape)
        # print("x16", x16.shape)
        # print("x32", x32.shape)
        return [x4, x8, x16, x32]


class FeatUp(SubModule):
    def __init__(self):
        super(FeatUp, self).__init__()
        chans = [16, 24, 32, 96, 160]
        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)
        self.deconv16_8 = Conv2x(chans[3] * 2, chans[2], deconv=True, concat=True)
        self.deconv8_4 = Conv2x(chans[2] * 2, chans[1], deconv=True, concat=True)
        self.conv4 = BasicConv(chans[1] * 2, chans[1] * 2, kernel_size=3, stride=1, padding=1)

        self.weight_init()

    def forward(self, featL, featR=None):
        x4, x8, x16, x32 = featL

        y4, y8, y16, y32 = featR
        x16 = self.deconv32_16(x32, x16)
        y16 = self.deconv32_16(y32, y16)

        x8 = self.deconv16_8(x16, x8)
        y8 = self.deconv16_8(y16, y8)
        x4 = self.deconv8_4(x8, x4)
        y4 = self.deconv8_4(y8, y4)
        x4 = self.conv4(x4)
        y4 = self.conv4(y4)

        return [x4, x8, x16, x32], [y4, y8, y16, y32]


class sr_context_upsample(nn.Module):
    def __init__(self, num_block=10):
        super(sr_context_upsample, self).__init__()
        self.sr4_1 = CLKA_Refinement(num_in_ch=9, num_out_ch=9, scale=4, num_feat=32,
                                     num_block=num_block, d_atten=64, conv_groups=2)

    def forward(self, depth_low, up_weights):
        b, c, h, w = depth_low.shape

        depth_unfold = F.unfold(depth_low.reshape(b, c, h, w), 3, 1, 1).reshape(b, -1, h, w)
        depth_unfold = self.sr4_1(depth_unfold)

        depth_unfold = depth_unfold.reshape(b, 9, h * 4, w * 4)
        depth = (depth_unfold * up_weights).sum(1)

        return depth


class nextstereo(nn.Module):
    def __init__(self, maxdisp=192, train_refine=False):
        super(nextstereo, self).__init__()
        self.maxdisp = maxdisp
        self.feature = Feature()
        self.feature_up = FeatUp()
        self.train_refine = train_refine

        self.stem_2 = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU()
        )
        self.stem_4 = nn.Sequential(
            BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU()
        )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1), )
        self.spx_2 = Conv2x(32, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv(96, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU()
        )

        self.conv = BasicConv(96, 48, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(48, 48, kernel_size=1, padding=0, stride=1)
        self.semantic = nn.Sequential(
            BasicConv(96, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 8, kernel_size=1, padding=0, stride=1, bias=False))
        self.agg = BasicConv(8, 8, is_3d=True, kernel_size=(1, 5, 5), padding=(0, 2, 2), stride=1)
        self.AggNeXt = AggNeXt(block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2], exp_r=3, kernel_size=5)
        self.corr_stem = BasicConv(1, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        if self.train_refine:
            self.sr_context_upsample = sr_context_upsample(num_block=20)

    def forward(self, left, right):
        features_left = self.feature(left)
        features_right = self.feature(right)
        features_left, features_right = self.feature_up(features_left, features_right)
        stem_2x = self.stem_2(left)
        stem_4x = self.stem_4(stem_2x)
        stem_2y = self.stem_2(right)
        stem_4y = self.stem_4(stem_2y)

        features_left[0] = torch.cat((features_left[0], stem_4x), 1)
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)

        match_left = self.desc(self.conv(features_left[0]))
        match_right = self.desc(self.conv(features_right[0]))

        corr_volume = build_norm_correlation_volume(match_left, match_right, self.maxdisp // 4)
        corr_volume = self.corr_stem(corr_volume)
        feat_volume = self.semantic(features_left[0]).unsqueeze(2)
        volume = self.agg(feat_volume * corr_volume)
        cost = self.AggNeXt(volume, features_left)

        xspx = self.spx_4(features_left[0])
        xspx = self.spx_2(xspx, stem_2x)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)

        disp_samples = torch.arange(0, self.maxdisp // 4, dtype=cost.dtype, device=cost.device)
        disp_samples = disp_samples.view(1, self.maxdisp // 4, 1, 1).repeat(cost.shape[0], 1, cost.shape[3],
                                                                            cost.shape[4])
        pred = regression_topk(cost.squeeze(1), disp_samples, 2)
        if self.train_refine:
            pred_up = self.sr_context_upsample(pred, spx_pred)
        else:
            pred_up = context_upsample(pred, spx_pred)

        if self.training:
            return [pred_up * 4, pred.squeeze(1) * 4]

        else:
            return [pred_up * 4]


if __name__ == '__main__':
    model = nextstereo().cuda()
    inputs1 = torch.randn(1, 3, 320, 1216).cuda()
    inputs2 = torch.randn(1, 3, 320, 1216).cuda()
    outputs = model(inputs1, inputs2)
    print("outputs:", outputs[0].shape)
