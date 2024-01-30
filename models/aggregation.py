import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.FeatureExtraction import Conv

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

class newchannelAtt(SubModule):
    def __init__(self, cv_chan, im_chan):
        super(newchannelAtt, self).__init__()

        self.im_att = nn.Sequential(
            # BasicConv(im_chan, im_chan // 2, kernel_size=1, stride=1, padding=0),
            Conv(im_chan, im_chan // 2, 1, 1),
            nn.Conv2d(im_chan // 2, cv_chan, 1),
            # Conv(im_chan, cv_chan, 3, 1)
        )

        self.weight_init()

    def forward(self, cv, im):
        channel_att = self.im_att(im)
        cv = torch.sigmoid(channel_att) * cv
        return cv


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class SimpleBottleneck(nn.Module):
    """Simple bottleneck block without channel expansion"""

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(SimpleBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Adaptive intra-scale aggregation & adaptive cross-scale aggregation
class AdaptiveAggregationModule(nn.Module):
    def __init__(self, num_scales, num_output_branches, max_disp=24,
                 num_blocks=1,
                 simple_bottleneck=False,
                 deformable_groups=2,
                 mdconv_dilation=2,
                 channelAtt=28):
        super(AdaptiveAggregationModule, self).__init__()

        self.num_scales = num_scales
        self.num_output_branches = num_output_branches
        self.max_disp = max_disp
        self.num_blocks = num_blocks

        self.branches = nn.ModuleList()

        # Adaptive intra-scale aggregation
        for i in range(self.num_scales):
            num_candidates = max_disp // (2 ** i)
            branch = nn.ModuleList()
            for j in range(num_blocks):
                if simple_bottleneck:
                    branch.append(SimpleBottleneck(num_candidates, num_candidates))
                else:
                    branch.append(SimpleBottleneck(num_candidates, num_candidates))

            self.branches.append(nn.Sequential(*branch))

        self.fuse_layers = nn.ModuleList()

        # 自适应跨尺度聚合
        for i in range(self.num_output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.num_scales):
                if i == j:
                    # Identity
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    self.fuse_layers[-1].append(
                        nn.Sequential(nn.Conv2d(max_disp // (2 ** j), max_disp // (2 ** i),
                                                kernel_size=1, bias=False),
                                      nn.BatchNorm2d(max_disp // (2 ** i)),
                                      ))
                elif i > j:
                    layers = nn.ModuleList()
                    for k in range(i - j - 1):
                        layers.append(nn.Sequential(nn.Conv2d(max_disp // (2 ** j), max_disp // (2 ** j),
                                                              kernel_size=3, stride=2, padding=1, bias=False),
                                                    nn.BatchNorm2d(max_disp // (2 ** j)),
                                                    nn.LeakyReLU(0.2, inplace=True),
                                                    ))

                    layers.append(nn.Sequential(nn.Conv2d(max_disp // (2 ** j), max_disp // (2 ** i),
                                                          kernel_size=3, stride=2, padding=1, bias=False),
                                                nn.BatchNorm2d(max_disp // (2 ** i))))
                    self.fuse_layers[-1].append(nn.Sequential(*layers))

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.feature_att_up_4 = newchannelAtt(channelAtt, 96)
        # self.feature_att_up_8 = newchannelAtt(14, 64)
        # self.feature_att_up_16 = newchannelAtt(7, 192)

    def forward(self, x, imgs):
        assert len(self.branches) == len(x)

        for i in range(len(self.branches)):
            branch = self.branches[i]
            for j in range(self.num_blocks):
                dconv = branch[j]
                x[i] = dconv(x[i])

        if self.num_scales == 1:  # without fusions
            return x

        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    exchange = self.fuse_layers[i][j](x[j])
                    if exchange.size()[2:] != x_fused[i].size()[2:]:
                        exchange = F.interpolate(exchange, size=x_fused[i].size()[2:],
                                                 mode='bilinear', align_corners=False)
                    x_fused[i] = x_fused[i] + exchange
        # print("x_fused[0]", x_fused[0].shape)
        # print("imgs[1]", imgs[1].shape)
        x_fused[0] = self.feature_att_up_4(x_fused[0], imgs[0])
        # x_fused[1] = self.feature_att_up_8(x_fused[1], imgs[1])
        # x_fused[2] = self.feature_att_up_16(x_fused[2], imgs[2])

        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])
            # print("x_fused", i, x_fused[i].shape)

        return x_fused

# Stacked AAModules
class AdaptiveAggregation(nn.Module):
    def __init__(self, max_disp=48, num_scales=3, num_fusions=6,
                 num_stage_blocks=1,
                 num_deform_blocks=2,
                 intermediate_supervision=True,
                 deformable_groups=2,
                 mdconv_dilation=2,
                 channelAtt=48):
        super(AdaptiveAggregation, self).__init__()

        self.max_disp = max_disp
        self.num_scales = num_scales
        self.num_fusions = num_fusions
        self.intermediate_supervision = intermediate_supervision

        fusions = nn.ModuleList()
        for i in range(num_fusions):
            if self.intermediate_supervision:
                num_out_branches = self.num_scales
            else:
                num_out_branches = 1 if i == num_fusions - 1 else self.num_scales

            if i >= num_fusions - num_deform_blocks:
                simple_bottleneck_module = False
            else:
                simple_bottleneck_module = True

            fusions.append(AdaptiveAggregationModule(num_scales=self.num_scales,
                                                     num_output_branches=num_out_branches,
                                                     max_disp=max_disp,
                                                     num_blocks=num_stage_blocks,
                                                     mdconv_dilation=mdconv_dilation,
                                                     deformable_groups=deformable_groups,
                                                     simple_bottleneck=simple_bottleneck_module,
                                                     channelAtt=channelAtt))

        self.fusions = nn.Sequential(*fusions)

        self.final_conv = nn.ModuleList()
        for i in range(self.num_scales):
            in_channels = max_disp // (2 ** i)

            self.final_conv.append(nn.Conv2d(in_channels, max_disp // (2 ** i), kernel_size=1))

            if not self.intermediate_supervision:
                break

    def forward(self, cost_volume, imgs):
        assert isinstance(cost_volume, list)

        for i in range(self.num_fusions):
            fusion = self.fusions[i]
            cost_volume = fusion(cost_volume, imgs)

        # Make sure the final output is in the first position
        out = []  # 1/3, 1/6, 1/12
        for i in range(len(self.final_conv)):
            out = out + [self.final_conv[i](cost_volume[i])]

        return out


if __name__ == '__main__':
    import time
    import torch
    import torchsummary
    # import thop
    import numpy as np

    input2 = torch.FloatTensor(1, 24, 160, 608).cuda()
    input = []
    input.append(input2)
    input1 = torch.FloatTensor(1, 24, 160, 608).cuda()
    # input1 = [input1]
    model = AdaptiveAggregation(max_disp=24,
                                               num_scales=1,
                                               num_fusions=6,
                                               num_stage_blocks=2,
                                               num_deform_blocks=3,
                                               mdconv_dilation=2,
                                               deformable_groups=2,
                                               intermediate_supervision=not False).cuda()
    output = model(input)

    # 计算参数量
    # input_size = np.array([(32, 160, 608), (32, 160, 608)])
    # summary = torchsummary.summary(model, input_size=input_size)

    # 计算计算量
    # flops, params = thop.profile(model, inputs=[input])
    # print(f'参数量: {params / 1e6:.2f}M')
    # print(f'计算量: {flops / 1e9:.2f}G FLOPs')

    total_time = 0
    for i in range(13):
        start_time = time.time()
        out = model(input)
        elapsed_time = (time.time() - start_time) * 1000
        total_time += elapsed_time
        print('第 %d 次执行的时间为：%.2f 毫秒' % (i + 1, elapsed_time))

    average_time = total_time / 10
    print('平均耗时为：%.2f 毫秒' % average_time)