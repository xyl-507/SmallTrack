import torch
import torch.nn as nn
# from DWT import DWT_2D
from siamban.models.backbone.DWT.DWT_layer import DWT_1D, IDWT_1D, DWT_2D_tiny, DWT_2D, IDWT_2D
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import pywt
import imageio

def show_tensor(a: torch.Tensor, fig_num=None, title=None):
    """Display a 2D tensor.
    args:
        fig_num: Figure number.
        title: Title of figure.
    """
    a_np = a.squeeze().cpu().clone().detach().numpy()
    # a_np = a
    if a_np.ndim == 3:
        a_np = np.transpose(a_np, (1, 2, 0))
    plt.figure(fig_num)
    plt.tight_layout()
    # plt.tight_layout(pad=0)  # 也可以设为默认的，但这样图大点
    plt.cla()
    plt.imshow(a_np)
    plt.axis('off')
    plt.axis('equal')
    # plt.colorbar()  # 创建颜色条
    if title is not None:
        plt.title(title)
    plt.draw()
    # plt.show()  # imshow是对图像的处理，show是展示图片
    plt.pause(0.1)

class wad_module(nn.Module):
    '''
    This module is used in directly connected networks.
    X --> output
    Args:
        wavename: Wavelet family
    '''
    def __init__(self, wavename='haar'):  # wavelist() or [‘haar’, ‘db’, ‘sym’, ‘coif’, ‘bior’, ‘rbio’, ‘dmey’]
        super(wad_module, self).__init__()
        self.dwt = DWT_2D(wavename=wavename)
        self.softmax = nn.Softmax2d()

    @staticmethod
    def get_module_name():
        return "wad"

    def forward(self, input):
        LL, LH, HL, _ = self.dwt(input)

        # lam = 1  # soft_threshold
        # LH=softthresholding(LH,lam)
        # HL=softthresholding(HL,lam)
        # _=softthresholding(_,lam)


        # for ll, lh, hl, hh in zip(LL, LH, HL, _):  # 可视化 x,  xyl 20221004
        #     ll = ll.sum(dim=0)
        #     plt.subplot(2, 2, 1)
        #     show_tensor(ll, 1, 'LL')
        #     lh = lh.sum(dim=0)
        #     plt.subplot(2, 2, 2)
        #     show_tensor(lh, 1, 'LH')
        #     hl = hl.sum(dim=0)
        #     plt.subplot(2, 2, 3)
        #     show_tensor(hl, 1, 'HL')
        #     hh = hh.sum(dim=0)
        #     plt.subplot(2, 2, 4)
        #     show_tensor(hh, 1, 'HH')

        output = LL
        # output = LL.sub(_)  # xyl 20221013 想减去噪声信息 HH

        x_high = self.softmax(torch.add(LH, HL))
        # x_high = self.softmax(torch.add(LH, HL).sub(_))  # xyl 20221013 想减去噪声信息 HH
        AttMap = torch.mul(output, x_high)
        output = torch.add(output, AttMap)
        # for k in x_high:  # 可视化 xyl 20221011
        #     show_tensor(k.sum(dim=0),3, 'x_high')
        # for k in AttMap:
        #     show_tensor(k.sum(dim=0),2, 'AttMap')
        # for k in output:
        #     show_tensor(k.sum(dim=0),1, 'haar')
        return output

class wad_module2(nn.Module):
    '''
    This module is used in directly connected networks.
    X --> output
    Args:
        wavename: Wavelet family
    '''
    def __init__(self, wavename='haar'):  # wavelist() or [‘haar’, ‘db’, ‘sym’, ‘coif’, ‘bior’, ‘rbio’, ‘dmey’]
        super(wad_module2, self).__init__()
        self.dwt = DWT_2D(wavename=wavename)
        self.softmax = nn.Softmax2d()

    @staticmethod
    def get_module_name():
        return "wad"

    def forward(self, input):
        LL, LH, HL, _ = self.dwt(input)

        # for ll, lh, hl, hh in zip(LL, LH, HL, _):  # 可视化 x,  xyl 20221004
        #     ll = ll.sum(dim=0)
        #     plt.subplot(2, 2, 1)
        #     show_tensor(ll, 1, 'LL')
        #     lh = lh.sum(dim=0)
        #     plt.subplot(2, 2, 2)
        #     show_tensor(lh, 1, 'LH')
        #     hl = hl.sum(dim=0)
        #     plt.subplot(2, 2, 3)
        #     show_tensor(hl, 1, 'HL')
        #     hh = hh.sum(dim=0)
        #     plt.subplot(2, 2, 4)
        #     show_tensor(hh, 1, 'HH')


        output = LL

        x_high = self.softmax(torch.add(LH, HL))
        AttMap = torch.mul(output, x_high)
        output = torch.add(output, AttMap)
        # for k in x_high:  # 可视化 xyl 20221011
        #     show_tensor(k.sum(dim=0))
        # for k in AttMap:
        #     show_tensor(k.sum(dim=0))
        for k in output:
            show_tensor(k.sum(dim=0), 2, 'haar')
        return output

class wad_module3(nn.Module):
    '''
    This module is used in directly connected networks.
    X --> output
    Args:
        wavename: Wavelet family
    '''
    def __init__(self, wavename='bior2.2'):  # wavelist() or [‘haar’, ‘db’, ‘sym’, ‘coif’, ‘bior’, ‘rbio’, ‘dmey’]
        super(wad_module3, self).__init__()
        self.dwt = DWT_2D(wavename=wavename)
        self.softmax = nn.Softmax2d()

    @staticmethod
    def get_module_name():
        return "wad"

    def forward(self, input):
        LL, LH, HL, _ = self.dwt(input)

        # for ll, lh, hl, hh in zip(LL, LH, HL, _):  # 可视化 x,  xyl 20221004
        #     ll = ll.sum(dim=0)
        #     plt.subplot(2, 2, 1)
        #     show_tensor(ll, 1, 'LL')
        #     lh = lh.sum(dim=0)
        #     plt.subplot(2, 2, 2)
        #     show_tensor(lh, 1, 'LH')
        #     hl = hl.sum(dim=0)
        #     plt.subplot(2, 2, 3)
        #     show_tensor(hl, 1, 'HL')
        #     hh = hh.sum(dim=0)
        #     plt.subplot(2, 2, 4)
        #     show_tensor(hh, 1, 'HH')

        output = LL

        x_high = self.softmax(torch.add(LH, HL))
        AttMap = torch.mul(output, x_high)
        output = torch.add(output, AttMap)
        # for k in x_high:  # 可视化 xyl 20221011
        #     show_tensor(k.sum(dim=0))
        # for k in AttMap:
        #     show_tensor(k.sum(dim=0))
        for k in output:
            show_tensor(k.sum(dim=0),3, 'bior2.2')
        return output

# Soft Thresholding 太慢了
def softthresholding(b, lam):
    m = nn.Softshrink(lambd=lam)  # 还有 Hardshrink
    soft_thresh = m(b)
    return soft_thresh.cuda()

if __name__ == '__main__':
    net = wad_module()
    # net.eval()

    # 会报错： RuntimeError: "addmm_cuda" not implemented for 'Byte'
    img = imageio.imread(r'D:\XYL\3.Object tracking\0.创新\1.小波变换等\WaveletAttention-main-小波变换注意力\noise\dog-guassian.jpg')  # array (255,255,3)
    img1 = np.transpose(img, (2, 0, 1))  # array (3, 255, 255)
    img2 = torch.FloatTensor(img1)  # tensor (3, 255, 255)
    img3 = img2.unsqueeze(dim=0)  # tensor (1, 3, 255, 255)
    out = net(img3)

    # var = torch.randn(1, 256, 31, 31)  # pysot 下采样之前的维度
    # # var = pywt.data.camera().astype(np.float32)
    # out = net(var)

    print(out[0].shape)
