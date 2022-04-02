# Implementation of attention mechanisms in a plug-and-play manner
# Implemented by Yuchuan Li

import torch
import torch.nn as nn


# spatial attentions
class SelfAttention(nn.Module):
    # self attention module
    # from https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

    def __init__(self, channel):
        super(SelfAttention, self).__init__()
        self.chanel_in = channel
        self.query = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.key = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.value = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        # Q
        proj_query = self.query(x).reshape(m_batchsize, -1, width * height).permute(0, 2, 1)
        # K
        proj_key = self.key(x).reshape(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        # attention
        attention = self.softmax(energy)
        # V
        proj_value = self.value(x).reshape(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.reshape(m_batchsize, C, height, width)

        return out


class NonLocalAttention(nn.Module):
    # non-local attention
    # from https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.html
    def __init__(self, channel):
        super(NonLocalAttention, self).__init__()
        self.inter_channel = channel // 2
        # convolution phi
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                  padding=0, bias=False)
        # convolution theta
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                    padding=0, bias=False)
        # convolution g
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        # convolution for attention mask
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1,
                                   padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W], phi calculation
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2], theta calculation
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W], theta_phi calculation
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out


class h_sigmoid(nn.Module):
    # h_sigmoid function in coord attention
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    # h_switch function in coord attention
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttention(nn.Module):
    # coord attention
    # from https://openaccess.thecvf.com/content/CVPR2021/html/Hou_Coordinate_Attention_for_Efficient_Mobile_Network_Design_CVPR_2021_paper.html
    def __init__(self, in_channel, out_channel, reduction=32):
        super(CoordAttention, self).__init__()
        # global average pooling
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        # channel reduction
        mip = max(8, in_channel // reduction)

        self.conv1 = nn.Conv2d(in_channel, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, out_channel, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        # global average pooling
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # attention calculation
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


# end of CoordAttention


# channel attentions
class SEAttention(nn.Module):
    # SE attention
    # from https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html
    def __init__(self, channel, reduction=16):
        super(SEAttention, self).__init__()
        # gloabl average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SKAttention(nn.Module):
    # SK attention
    # from http://openaccess.thecvf.com/content_CVPR_2019/html/Li_Selective_Kernel_Networks_CVPR_2019_paper.html
    def __init__(self, channel, WH, M, G, r, stride=1, L=32):
        super(SKAttention, self).__init__()
        d = max(int(channel / r), L)
        self.M = M
        self.features = channel
        self.convs = nn.ModuleList([])
        for i in range(M):
            # convolution with different size of kernal
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(channel,
                              channel,
                              kernel_size=3 + i * 2,
                              stride=stride,
                              padding=1 + i,
                              groups=G), nn.BatchNorm2d(channel),
                    nn.ReLU(inplace=False)))

        self.fc = nn.Linear(channel, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, channel))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        # feature U
        fea_U = torch.sum(feas, dim=1)
        # feature s
        fea_s = fea_U.mean(-1).mean(-1)
        # feature z
        fea_z = self.fc(fea_s)
        # calculation for attention vectors
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


class GCTAttention(nn.Module):
    # GCT attention
    # from http://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Gated_Channel_Transformation_for_Visual_Recognition_CVPR_2020_paper.html
    def __init__(self, channel, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCTAttention, self).__init__()
        # alpha, beta and gamma matrix
        self.alpha = torch.ones((1, channel, 1, 1)).cuda()
        self.gamma = torch.zeros((1, channel, 1, 1)).cuda()
        self.beta = torch.zeros((1, channel, 1, 1)).cuda()
        self.epsilon = epsilon
        # l2 normalization
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):
        # l2 normalization process
        if self.mode == 'l2':
            embedding = (x.pow(2).sum(2, keepdims=True).sum(3, keepdims=True) +
                         self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / \
                   (embedding.pow(2).mean(dim=1, keepdims=True) + self.epsilon).pow(0.5)

        # option for l1 normalization is also here for reference (although not used)
        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum(2, keepdims=True).sum(
                3, keepdims=True) * self.alpha
            norm = self.gamma / \
                   (torch.abs(embedding).mean(dim=1, keepdims=True) + self.epsilon)
        else:
            print('Unknown mode!')
        # gate value
        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate


# hybrid attention
class ChannelAttention(nn.Module):
    # channel attention branch for CBAM attention
    def __init__(self, channel, ratio=16):
        super(ChannelAttention, self).__init__()
        # global average pooling, which is similar to SE attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # a MLP with 2 conv layer
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # average output
        avg_out = self.shared_MLP(self.avg_pool(x))
        # maximum output
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    # spatial attention branch for CBAM attention
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # check the kernal size definition
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # average output
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # maximum output
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAMAttention(nn.Module):
    # CBAM attention
    # from http://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.html
    def __init__(self, channel):
        super(CBAMAttention, self).__init__()
        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


# end of CBAM


class PAModule(nn.Module):
    # Position attention module for DANet

    def __init__(self, channel):
        super(PAModule, self).__init__()
        self.chanel_in = channel
        # Q, K, V definition
        self.query_conv = nn.Conv2d(in_channels=channel, out_channels=channel // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=channel, out_channels=channel // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.gamma = torch.zeros(1).cuda()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        # the overall calculation is similar to Self-attention
        m_batchsize, C, height, width = x.size()
        # Q
        proj_query = self.query_conv(x).reshape(m_batchsize, -1, width * height).permute(0, 2, 1)
        # K
        proj_key = self.key_conv(x).reshape(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        # attention
        attention = self.softmax(energy)
        # V
        proj_value = self.value_conv(x).reshape(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.reshape(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAModule(nn.Module):
    # Channel attention module for DANet

    def __init__(self, channel):
        super(CAModule, self).__init__()
        self.chanel_in = channel
        self.gamma = torch.zeros(1).cuda()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        # the overall calculation is similar to Self-attention
        m_batchsize, C, height, width = x.size()
        # Q
        proj_query = x.reshape(m_batchsize, C, -1)
        # K
        proj_key = x.reshape(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        # attention
        attention = self.softmax(energy)
        # V
        proj_value = x.reshape(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.reshape(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class DANetAttention(nn.Module):
    # DANet attention
    # from http://openaccess.thecvf.com/content_CVPR_2019/html/Fu_Dual_Attention_Network_for_Scene_Segmentation_CVPR_2019_paper.html
    def __init__(self, in_channel, out_channel):
        super(DANetAttention, self).__init__()
        inter_channels = in_channel // 4
        # convolution for spatial
        self.conv5a = nn.Sequential(nn.Conv2d(in_channel, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        # convolution for channel-wise
        self.conv5c = nn.Sequential(nn.Conv2d(in_channel, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.sa = PAModule(inter_channels)
        self.sc = CAModule(inter_channels)

        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.conv8 = nn.Sequential(nn.Dropout(0.1, False), nn.Conv2d(inter_channels, out_channel, 1))

    def forward(self, x):
        # calculation of spatial attention
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)

        # calculation of channel attention
        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)

        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)

        return sasc_output


# end of DANetAttention


class BasicConv(nn.Module):
    # basic convolution block in triplet attention
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            relu=True,
            bn=True,
            bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_channel
        self.conv = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        # batch normalization
        self.bn = (
            nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    # Z-Pool in triplet attention
    def forward(self, x):
        y, _ = torch.max(x, 1)
        y = y.unsqueeze(1)
        return torch.concat((y, torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    # attention gate implementation for triplet attention
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        # compression and convolution block definition
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        # compression, output feature and scaling
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = x_out.sigmoid()
        return x * scale


class TripletAttention(nn.Module):
    # triplet attention
    # from https://openaccess.thecvf.com/content/WACV2021/html/Misra_Rotate_to_Attend_Convolutional_Triplet_Attention_Module_WACV_2021_paper.html?ref=https://githubhelp.com
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        # attention gate definition
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3)
        x_out1 = self.cw(x_perm1)
        # output from gate 1
        x_out11 = x_out1.permute(0, 2, 1, 3)
        x_perm2 = x.permute(0, 3, 2, 1)
        x_out2 = self.hc(x_perm2)
        # output from gate 2
        x_out21 = x_out2.permute(0, 3, 2, 1)
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out

# end of TripletAttention
