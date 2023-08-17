import math

import torch.nn as nn
import torch
from siamban.models.backbone.DWT.wad_module import wad_module

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50']


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        padding = 2 - stride

        if dilation > 1:
            padding = dilation

        dd = dilation
        pad = padding
        if downsample is not None and dilation > 1:
            dd = dilation // 2
            pad = dd

        self.conv1 = nn.Conv2d(inplanes, planes,
                               stride=stride, dilation=dd, bias=False,
                               kernel_size=3, padding=pad)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)  # 1*1 convolution
        # self.conv1 = GhostModule(inplanes, planes, kernel_size=1, bias=False)  # ghost module xyl 20210204
        self.bn1 = nn.BatchNorm2d(planes)
        padding = 2 - stride  # ----------------------
        if downsample is not None and dilation > 1:
            dilation = dilation // 2
            padding = dilation

        assert stride == 1 or dilation == 1, \
            "stride and dilation must have one equals to zero at least"

        if dilation > 1:
            padding = dilation  # -----------------这之间的和原版resnet的区别 ，对padding的设置
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,  # 3*3 convolution
                               padding=padding, bias=False, dilation=dilation)  # padding可能会不一样
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)  # 1*1 convolution
        self.bn3 = nn.BatchNorm2d(planes * 4)
        # self.se = SELayer(planes*4, reduction)  # xyl 20210203
        # self.sa = sa_layer(planes*4, reduction)  # xyl 20210203
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)


        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, used_layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0,  # 3  --------没有padding=3
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # xyl 20210203 试着换成softpool
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1],
                                       stride=2)  # 原始resnet的layers234的stride都为2，这里的sride也是右路downsample的strid，要改动也是这个为1！！！

        self.feature_size = 128 * block.expansion
        self.used_layers = used_layers
        layer3 = True if 3 in used_layers else False  # config文件uesd layer如果是123，则4就不用运行了，直接输出上一层的输入，毕竟RPN没有连上4
        layer4 = True if 4 in used_layers else False

        if layer3:
            self.layer3 = self._make_layer(block, 256, layers[2],
                                           stride=1, dilation=2)  # 15x15, 7x7  -----原stride=2，没有dilation
            self.feature_size = (256 + 128) * block.expansion
        else:
            self.layer3 = lambda x: x  # identity

        if layer4:
            self.layer4 = self._make_layer(block, 512, layers[3],
                                           stride=1, dilation=4)  # 7x7, 3x3
            self.feature_size = 512 * block.expansion
        else:
            self.layer4 = lambda x: x  # identity

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.wad = wad_module()

    # self.layer1 = self._make_layer(block, 64, layers[0])  3
    # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  4
    # self.layer3 = self._make_layer(block, 256, layers[2],stride=1, dilation=2)  6
    # self.layer4 = self._make_layer(block, 512, layers[3],stride=1, dilation=4)  3

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):  # 多了dilation
        downsample = None  # ----------------------------开始
        dd = dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1 and dilation == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                if dilation > 1:
                    dd = dilation // 2
                    padding = dd
                    # downsample = nn.Sequential(
                    #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    #     nn.Conv2d(self.inplanes, planes * block.expansion,
                    #               kernel_size=1, stride=1, bias=False,
                    #               # 原版resnet用的是1*1，stride为2的卷积，按这种3*3的情况来说改动不了
                    #               padding=padding, dilation=dd),
                    #     nn.BatchNorm2d(planes * block.expansion),
                    # )

                else:  # ------------------------结束，多出来的部分
                    # reduce the effective strides at the last two block from 16 pixels and 32 pixels to 8 pixels by modifying the conv4 and conv5 block to have unit spatial stride,
                    # and also increase its receptive ﬁeld by dilated convolutions
                    dd = 1
                    padding = 0
                    # downsample = nn.Sequential(
                    #     nn.Conv2d(self.inplanes, planes * block.expansion,
                    #               kernel_size=3, stride=stride, bias=False,
                    #               # 原版resnet用的是1*1，stride为2的卷积，按这种3*3的情况来说改动不了
                    #               padding=padding, dilation=dd),
                    #     nn.BatchNorm2d(planes * block.expansion),
                    # )
                # if stride == 2:    #  DWT xyl 20221011
                #     downsample = nn.Sequential(
                #         wad_module(),
                #         nn.Conv2d(self.inplanes, planes * block.expansion,
                #                 kernel_size=1, stride=1, bias=False,  # 原版resnet用的是1*1，stride为2的卷积，按这种3*3的情况来说改动不了
                #                 padding=padding, dilation=dd),
                #         nn.BatchNorm2d(planes * block.expansion),
                #     )
                # else:
                #     downsample = nn.Sequential(
                #         nn.Conv2d(self.inplanes, planes * block.expansion,
                #                 kernel_size=3, stride=stride, bias=False,  # 原版resnet用的是1*1，stride为2的卷积，按这种3*3的情况来说改动不了
                #                 padding=padding, dilation=dd),
                #         nn.BatchNorm2d(planes * block.expansion),
                #     )

                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=3, stride=stride, bias=False,  # 原版resnet用的是1*1，stride为2的卷积，按这种3*3的情况来说改动不了
                              padding=padding, dilation=dd),
                    nn.BatchNorm2d(planes * block.expansion),
                )


        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            downsample, dilation=dilation))
        self.inplanes = planes * block.expansion  # ------------------------------------------
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x_ = self.relu(x)
        # x1 = self.maxpool(x_)
        x = self.wad(x_)  # DWT
        # x = x1 + x
        p1 = self.layer1(x)
        # p1g = self.maxpool(p1)
        p2 = self.layer2(p1)
        p3 = self.layer3(p2)
        p4 = self.layer4(p3)
        out = [x_, p1, p2, p3, p4]  # 使用的是池化之前的特征层，why？
        # out = [x_, p1g, p2, p3, p4]  # xyl 20210203 改变layer1的维度，使其能和layer 2 3相加
        out = [out[i] for i in self.used_layers]  # 根据used_layer来设置输出，xyl 20210202
        if len(out) == 1:
            return out[0]
        else:
            return out


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.

    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.

    """
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   **kwargs)  # [3,4,6,3]对应于ResNet50中 conv2_x中 有三个（1*1*64,3*3*64,1*1*256）卷积层的堆叠所以第一个数字为3
    return model


if __name__ == '__main__':
    net = resnet50()  # 大致与原版resnet一样，参考  https://blog.csdn.net/a940902940902/article/details/83858694
    # print(net)
    net = net.cuda()

    var = torch.FloatTensor(1, 3, 127, 127).cuda()
    out1 = net(var)
    print('out1.shape', out1.shape)

    var = torch.FloatTensor(1, 3, 255, 255).cuda()
    # var = Variable(var)
    out2 = net(var)
    print('out2.shape', out2.shape)
