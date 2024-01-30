# VAst-receptive-field Pixel attention network
import torch.nn as nn
from basicsr.archs.arch_util import default_init_weights

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pointwise = nn.Conv2d(dim, dim, 1)
        self.depthwise = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.depthwise_dilated = nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)

    def forward(self, x):
        u = x.clone()
        attn = self.pointwise(x)
        attn = self.depthwise(attn)
        attn = self.depthwise_dilated(attn)
        return u * attn

class VAB(nn.Module):
    def __init__(self, d_model, d_atten):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_atten, 1)
        self.activation = nn.GELU()
        self.atten_branch = Attention(d_atten)
        self.proj_2 = nn.Conv2d(d_atten, d_model, 1)
        self.pixel_norm = nn.LayerNorm(d_model)
        default_init_weights([self.pixel_norm], 0.1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.atten_branch(x)
        x = self.proj_2(x)
        x = x + shorcut

        x = x.permute(0, 2, 3, 1) #(B, H, W, C)
        x = self.pixel_norm(x)
        x = x.permute(0, 3, 1, 2).contiguous() #(B, C, H, W)

        return x

def pixelshuffle(in_channels, out_channels, upscale_factor=4):
    upconv1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(2)
    upconv2 = nn.Conv2d(16, out_channels * 4, 3, 1, 1)
    lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    return nn.Sequential(*[upconv1, pixel_shuffle, lrelu, upconv2, pixel_shuffle])

#both scale X2 and X3 use this version
def pixelshuffle_single(in_channels, out_channels, upscale_factor=2):
    upconv1 = nn.Conv2d(in_channels, 56, 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    upconv2 = nn.Conv2d(56, out_channels * upscale_factor * upscale_factor, 3, 1, 1)
    lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    return nn.Sequential(*[upconv1, lrelu, upconv2, pixel_shuffle])


def make_layer(block, n_layers, *kwargs):
    layers = []
    for _ in range(n_layers):
        layers.append(block(*kwargs))
    return nn.Sequential(*layers)


class CLKA_Refinement(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, d_atten=64, conv_groups=1):
        super(CLKA_Refinement, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(VAB, num_block, num_feat, d_atten)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1, groups=conv_groups) #conv_groups=2 for VapSR-S

        # upsample
        if scale == 4:
            self.upsampler = pixelshuffle(num_feat, num_out_ch, upscale_factor=scale)
        else:
            self.upsampler = pixelshuffle_single(num_feat, num_out_ch, upscale_factor=scale)

    def forward(self, feat):
        feat = self.conv_first(feat)
        body_feat = self.body(feat)
        body_out = self.conv_body(body_feat)
        feat = feat + body_out
        out = self.upsampler(feat)
        return out


if __name__ == '__main__':
    model = vapsr(num_in_ch=160, num_out_ch=96, scale=2, num_feat=32,
                  num_block=11, d_atten=64, conv_groups=2).cuda()

    model.train()

    import time
    import datetime
    import torch

    input = torch.FloatTensor(1, 160, 10, 38).zero_().cuda()
    out = model(input)
    print("out", out.shape)
    start = datetime.datetime.now()
    # for i in range(0):
    # pass
    # out = model(input, input)
    total = 0
    for i in range(1):
        start = datetime.datetime.now()

        out = model(input)
        # print('模型输出的长度:', len(out))
        # for i in range(len(out)):
        #     print("out", out[i].shape)

        end = datetime.datetime.now()
        print('本次调用时间为:', (end - start).total_seconds())
        total += (end - start).total_seconds()
    print('模型调用总时间为:', (total / 1))