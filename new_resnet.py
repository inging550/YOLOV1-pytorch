import torch
from torch.nn import Sequential, Conv2d, MaxPool2d, ReLU, BatchNorm2d
from torch import nn
from torch.utils import model_zoo

model_urls = {'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'}
CLASS_NUM = 20   # 使用其他训练集需要更改

class Bottleneck(nn.Module):  # 定义基本块
    def __init__(self, in_channel, out_channel, stride, downsample):
        super(Bottleneck, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.bottleneck = Sequential(

            Conv2d(in_channel, out_channel, kernel_size=1, stride=stride[0], padding=0, bias=False),
            BatchNorm2d(out_channel),
            ReLU(inplace=True),

            Conv2d(out_channel, out_channel, kernel_size=3, stride=stride[1], padding=1, bias=False),
            BatchNorm2d(out_channel),
            ReLU(inplace=True),

            Conv2d(out_channel, out_channel * 4, kernel_size=1, stride=stride[2], padding=0, bias=False),
            BatchNorm2d(out_channel * 4),
        )
        if self.downsample is False:  # 如果 downsample = True则为Conv_Block 为False为Identity_Block
            self.shortcut = Sequential()
        else:
            self.shortcut = Sequential(
                Conv2d(self.in_channel, self.out_channel * 4, kernel_size=1, stride=stride[0], bias=False),
                BatchNorm2d(self.out_channel * 4)
            )

    def forward(self, x):
        out = self.bottleneck(x)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class output_net(nn.Module):
    # no expansion
    # dilation = 2
    # type B use 1x1 conv
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_type='A'):
        super(output_net, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=2, bias=False, dilation=2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.downsample = nn.Sequential()
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_planes != self.expansion * planes or block_type == 'B':
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = self.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self, block):
        super(ResNet50, self).__init__()
        self.block = block
        self.layer0 = Sequential(
            Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self.make_layer(self.block, channel=[64, 64], stride1=[1, 1, 1], stride2=[1, 1, 1], n_re=3)
        self.layer2 = self.make_layer(self.block, channel=[256, 128], stride1=[2, 1, 1], stride2=[1, 1, 1], n_re=4)
        self.layer3 = self.make_layer(self.block, channel=[512, 256], stride1=[2, 1, 1], stride2=[1, 1, 1], n_re=6)
        self.layer4 = self.make_layer(self.block, channel=[1024, 512], stride1=[2, 1, 1], stride2=[1, 1, 1], n_re=3)
        self.layer5 = self._make_output_layer(in_channels=2048)
        self.avgpool = nn.AvgPool2d(2)  # kernel_size = 2  , stride = 2
        self.conv_end = nn.Conv2d(256, int(CLASS_NUM + 10), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_end = nn.BatchNorm2d(int(CLASS_NUM + 10))

    def make_layer(self, block, channel, stride1, stride2, n_re):
        layers = []
        for num_layer in range(0, n_re):
            if num_layer == 0:
                layers.append(block(channel[0], channel[1], stride1, downsample=True))
            else:
                layers.append(block(channel[1]*4, channel[1], stride2, downsample=False))
        return Sequential(*layers)

    def _make_output_layer(self, in_channels):
        layers = []
        layers.append(
            output_net(
                in_planes=in_channels,
                planes=256,
                block_type='B'))
        layers.append(
            output_net(
                in_planes=256,
                planes=256,
                block_type='A'))
        layers.append(
            output_net(
                in_planes=256,
                planes=256,
                block_type='A'))
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape) # 3*448*448
        out = self.layer0(x)
        # print(out.shape) # 64*112*112
        out = self.layer1(out)
        # print(out.shape)  # 256*112*112
        out = self.layer2(out)
        # print(out.shape) # 512*56*56
        out = self.layer3(out)
        # print(out.shape) # 1024*28*28
        out = self.layer4(out)  # 2048*14*14
        out = self.layer5(out)  # batch_size*256*14*14
        out = self.avgpool(out)  # batch_size*256*7*7
        out = self.conv_end(out)  # batch_size*30*7*7
        out = self.bn_end(out)
        out = torch.sigmoid(out)
        out = out.permute(0, 2, 3, 1)  # bitch_size*7*7*30
        return out


def resnet50(pretrained=False):
    model = ResNet50(Bottleneck)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


if __name__=='__main__':
    model = resnet50()
    for i in model.state_dict().keys():
        print(i)
