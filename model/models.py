# Implementation of models that used in this experiment
# Implemented by Yuchuan Li


from visualizer import get_local

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from model.attention import SelfAttention, NonLocalAttention, SEAttention, SKAttention, GCTAttention, \
    CBAMAttention, DANetAttention, TripletAttention, CoordAttention

# global reference definition
__all__ = ['ResNet', 'resnet18_attention', 'resnet34_attention', 'resnet50_attention']

# pre-trained checkpoint URL
model_urls = {
    'resnet18_attention': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34_attention': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50_attention': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


class BasicBlock(nn.Module):
    # basic block definition, from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    # modified by Yuchuan Li
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, att_choice=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        # intilization of different attention mechanisms
        self.selfatt = SelfAttention(out_channels)
        self.nlatt = NonLocalAttention(out_channels)
        self.cordatt = CoordAttention(out_channels, out_channels)

        self.seatt = SEAttention(out_channels)
        self.skatt = SKAttention(out_channels, WH=1, M=2, G=1, r=2)
        self.gctatt = GCTAttention(out_channels)

        self.cbamatt = CBAMAttention(out_channels)
        self.danetatt = DANetAttention(out_channels, out_channels)
        self.tripletatt = TripletAttention()

        # choice of attention
        self.att_choice = att_choice

    @get_local('attn_map')
    def forward(self, input):
        residual = input
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        # attention plug-in place
        if self.att_choice == 'Self_Attention':
            x1 = self.selfatt(x)
            attn_map = x1 - x
            x = x1

        if self.att_choice == 'Non_Local_Attention':
            x1 = self.nlatt(x)
            attn_map = x1 - x
            x = x1

        if self.att_choice == 'Coord_Attention':
            x1 = self.cordatt(x)
            attn_map = x1 - x
            x = x1

        if self.att_choice == 'SE_Attention':
            x1 = self.seatt(x)
            attn_map = x1 - x
            x = x1

        if self.att_choice == 'SK_Attention':
            x1 = self.skatt(x)
            attn_map = x1 - x
            x = x1

        if self.att_choice == 'GCT_Attention':
            x1 = self.gctatt(x)
            attn_map = x1 - x
            x = x1

        if self.att_choice == 'CBAM_Attention':
            x1 = self.cbamatt(x)
            attn_map = x1 - x
            x = x1

        if self.att_choice == 'DANet_Attention':
            x1 = self.danetatt(x)
            attn_map = x1 - x
            x = x1

        if self.att_choice == 'Triplet_Attention':
            x1 = self.tripletatt(x)
            attn_map = x1 - x
            x = x1

        # att_map = torch.div(out1, out)

        if self.downsample:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x


class BottleNeck(nn.Module):
    # bottleneck block definition, from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    # modified by Yuchuan Li
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, att_choice=None):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        # intilization of different attention mechanisms
        self.selfatt = SelfAttention(out_channels)
        self.nlatt = NonLocalAttention(out_channels)
        self.cordatt = CoordAttention(out_channels, out_channels)

        self.seatt = SEAttention(out_channels)
        self.skatt = SKAttention(out_channels, WH=1, M=2, G=1, r=2)
        self.gctatt = GCTAttention(out_channels)

        self.cbamatt = CBAMAttention(out_channels)
        self.danetatt = DANetAttention(out_channels, out_channels)
        self.tripletatt = TripletAttention()

        # choice of attention
        self.att_choice = att_choice

    @get_local('attn_map')
    def forward(self, input):
        residual = input
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        # attention plug-in place
        if self.att_choice == 'Self_Attention':
            x1 = self.selfatt(x)
            attn_map = x1 - x
            x = x1

        if self.att_choice == 'Non_Local_Attention':
            x1 = self.nlatt(x)
            attn_map = x1 - x
            x = x1

        if self.att_choice == 'Coord_Attention':
            x1 = self.cordatt(x)
            attn_map = x1 - x
            x = x1

        if self.att_choice == 'SE_Attention':
            x1 = self.seatt(x)
            attn_map = x1 - x
            x = x1

        if self.att_choice == 'SK_Attention':
            x1 = self.skatt(x)
            attn_map = x1 - x
            x = x1

        if self.att_choice == 'GCT_Attention':
            x1 = self.gctatt(x)
            attn_map = x1 - x
            x = x1

        if self.att_choice == 'CBAM_Attention':
            x1 = self.cbamatt(x)
            attn_map = x1 - x
            x = x1

        if self.att_choice == 'DANet_Attention':
            x1 = self.danetatt(x)
            attn_map = x1 - x
            x = x1

        if self.att_choice == 'Triplet_Attention':
            x1 = self.tripletatt(x)
            attn_map = x1 - x
            x = x1

        # att_map = torch.div(out1, out)

        if self.downsample:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    # ResNet implementation, from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    def __init__(self, block, num_layer, n_classes=1000, input_channels=3, att_choice=None):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.att_choice = att_choice
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_layer[0])
        self.layer2 = self._make_layer(block, 128, num_layer[1], 2)
        self.layer3 = self._make_layer(block, 256, num_layer[2], 2)
        self.layer4 = self._make_layer(block, 512, num_layer[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(block.expansion * 512, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, block, out_channels, num_block, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, att_choice=self.att_choice))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_channels, out_channels, att_choice=self.att_choice))
        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet18_attention(pretrained=False, n_classes=9, att_choice=None, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        :param pretrained: If True, returns a model pre-trained on ImageNet
        :param att_choice: choice of attention mechanisms
        :param n_classes: number of classes in the dataset
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], n_classes=n_classes, att_choice=att_choice, **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet18_attention'])
        now_state_dict = model.state_dict()

        # modification of original checkpoint in loading
        avoid = ['fc.weight', 'fc.bias']
        for key in pretrained_state_dict.keys():
            if key in avoid or key not in now_state_dict.keys():
                continue
            if pretrained_state_dict[key].size() != now_state_dict[key].size():
                continue
            now_state_dict[key] = pretrained_state_dict[key]

        model.load_state_dict(now_state_dict)
    return model


def resnet34_attention(pretrained=False, n_classes=9, att_choice=None, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        :param pretrained: If True, returns a model pre-trained on ImageNet
        :param att_choice: choice of attention mechanisms
        :param n_classes: number of classes in the dataset
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], n_classes=n_classes, att_choice=att_choice, **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet34_attention'])
        now_state_dict = model.state_dict()

        # modification of original checkpoint in loading
        avoid = ['fc.weight', 'fc.bias']
        for key in pretrained_state_dict.keys():
            if key in avoid or key not in now_state_dict.keys():  #
                continue
            if pretrained_state_dict[key].size() != now_state_dict[key].size():
                continue
            now_state_dict[key] = pretrained_state_dict[key]

        model.load_state_dict(now_state_dict)
    return model


def resnet50_attention(pretrained=False, n_classes=9, att_choice=None, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        :param pretrained: If True, returns a model pre-trained on ImageNet
        :param att_choice: choice of attention mechanisms
        :param n_classes: number of classes in the dataset
    """
    model = ResNet(BottleNeck, [3, 4, 6, 3], n_classes=n_classes, att_choice=att_choice, **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50_attention'])
        now_state_dict = model.state_dict()

        # modification of original checkpoint in loading
        avoid = ['fc.weight', 'fc.bias']
        for key in pretrained_state_dict.keys():
            if key in avoid or key not in now_state_dict.keys():  #
                continue
            if pretrained_state_dict[key].size() != now_state_dict[key].size():
                continue
            now_state_dict[key] = pretrained_state_dict[key]

        model.load_state_dict(now_state_dict)
    return model


class SimpleNet(nn.Module):
    # SimpleNet for MedMNIST dataset
    def __init__(self, in_channels=3, num_classes=9, att_choice=None):
        super(SimpleNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

        # intilization of different attention mechanisms
        self.selfatt = SelfAttention(64)
        self.nlatt = NonLocalAttention(64)
        self.cordatt = CoordAttention(64, 64)

        self.seatt = SEAttention(64)
        self.skatt = SKAttention(64, WH=1, M=2, G=1, r=2)
        self.gctatt = GCTAttention(64)

        self.cbamatt = CBAMAttention(64)
        self.danetatt = DANetAttention(64, 64)
        self.tripletatt = TripletAttention()

        # choice of attention
        self.att_choice = att_choice

    @get_local('attn_map')
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # attention plug-in place
        if self.att_choice == 'Self_Attention':
            x1 = self.selfatt(x)
            attn_map = x1 - x
            x = x1

        if self.att_choice == 'Non_Local_Attention':
            x1 = self.nlatt(x)
            attn_map = x1 - x
            x = x1

        if self.att_choice == 'Coord_Attention':
            x1 = self.cordatt(x)
            attn_map = x1 - x
            x = x1

        if self.att_choice == 'SE_Attention':
            x1 = self.seatt(x)
            attn_map = x1 - x
            x = x1

        if self.att_choice == 'SK_Attention':
            x1 = self.skatt(x)
            attn_map = x1 - x
            x = x1

        if self.att_choice == 'GCT_Attention':
            x1 = self.gctatt(x)
            attn_map = x1 - x
            x = x1

        if self.att_choice == 'CBAM_Attention':
            x1 = self.cbamatt(x)
            attn_map = x1 - x
            x = x1

        if self.att_choice == 'DANet_Attention':
            x1 = self.danetatt(x)
            attn_map = x1 - x
            x = x1

        if self.att_choice == 'Triplet_Attention':
            x1 = self.tripletatt(x)
            attn_map = x1 - x
            x = x1

        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
