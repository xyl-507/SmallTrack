import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F

from siamban.models.gal.batchnorm import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d
# from models.include.deeplabv3plus_inc.modeling.aspp import build_aspp
# from models.include.deeplabv3plus_inc.modeling.decoder import build_decoder
# from models.include.deeplabv3plus_inc.modeling.backbone import resnet


### help functions ###
def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        lambda_rule = lambda epoch: opt.lr_gamma ** ((epoch+1) // opt.lr_decay_epochs)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,step_size=opt.lr_decay_iters, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_net(net, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    return net

def define_GALNet(num_labels, gpu_ids=[], input_channels=1, use_gal=True):
    net = GALDeepLabV3Plus(n_class=num_labels, input_channels=input_channels, use_gal=use_gal)
    return init_net(net, gpu_ids)


# Ref: https://github.com/jfzhang95/pytorch-deeplab-xception
def build_backbone(backbone, output_stride, BatchNorm, input_channels):
    if backbone == 'resnet':
        return resnet.ResNet50(output_stride, BatchNorm, num_ch=input_channels)
    else:
        raise NotImplementedError

# class GALDeepLabV3Plus(nn.Module):
#     def __init__(self, n_class=2, backbone='resnet', output_stride=16, sync_bn=True, freeze_bn=False, input_channels=1, use_gal=True):
#         super(GALDeepLabV3Plus, self).__init__()
#
#         if backbone == 'drn':
#             output_stride = 8
#
#         if sync_bn == True:
#             BatchNorm = SynchronizedBatchNorm2d
#         else:
#             BatchNorm = nn.BatchNorm2d
#
#         self.backbone = build_backbone(backbone, output_stride, BatchNorm, input_channels)
#         self.aspp = build_aspp(backbone, output_stride, BatchNorm)
#         self.decoder = build_decoder(n_class, backbone, BatchNorm)
#
#         self.use_gal = False
#         if use_gal:
#             print("Using GAL")
#             self.use_gal = True
#             self.gal = GAL(sync_bn=sync_bn, input_channels=2048)
#
#         if freeze_bn:
#             self.freeze_bn()
#
#     def forward(self, input):
#         input = input.float()
#
#         x, low_level_feat = self.backbone(input)
#
#         if self.use_gal:
#             x = self.gal(x)
#
#         x = self.aspp(x)
#
#         x = self.decoder(x, low_level_feat)
#         x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
#
#         return x
#
#     def freeze_bn(self):
#         for m in self.modules():
#             if isinstance(m, SynchronizedBatchNorm2d):
#                 m.eval()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.eval()


class GAL(nn.Module):
    def __init__(self, sync_bn=True, input_channels=256):
        super(GAL, self).__init__()
        self.input_channels = input_channels
        if sync_bn == True:
            BatchNorm1d = SynchronizedBatchNorm1d
            BatchNorm2d = SynchronizedBatchNorm2d
        else:
            BatchNorm1d = nn.BatchNorm1d
            BatchNorm2d = nn.BatchNorm2d

        self.edge_aggregation_func = nn.Sequential(
            nn.Linear(4, 1),
            BatchNorm1d(1),
            nn.ReLU(inplace=True),
        )
        self.vertex_update_func = nn.Sequential(
            nn.Linear(2 * input_channels, input_channels // 2),
            BatchNorm1d(input_channels // 2),
            nn.ReLU(inplace=True),
        )

        self.edge_update_func = nn.Sequential(
            nn.Linear(2 * input_channels, input_channels // 2),
            BatchNorm1d(input_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.update_edge_reduce_func = nn.Sequential(
            nn.Linear(4, 1),
            BatchNorm1d(1),
            nn.ReLU(inplace=True),
        )

        self.final_aggregation_layer = nn.Sequential(
            nn.Conv2d(input_channels + input_channels // 2, input_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
        )

        self._init_weight()

    def forward(self, input):

        x = input
        B, C, H, W = x.size()

        vertex = input
        edge = torch.stack(
            (
                torch.cat((input[:,:,-1:], input[:,:,:-1]), dim=2),
                torch.cat((input[:,:,1:], input[:,:,:1]), dim=2),
                torch.cat((input[:,:,:,-1:], input[:,:,:,:-1]), dim=3),
                torch.cat((input[:,:,:,1:], input[:,:,:,:1]), dim=3)
            ), dim=-1
        ) * input.unsqueeze(dim=-1)

        aggregated_edge = self.edge_aggregation_func(
            edge.reshape(-1, 4)
        ).reshape((B, C, H, W))
        cat_feature_for_vertex = torch.cat((vertex, aggregated_edge), dim=1)
        update_vertex = self.vertex_update_func(
            cat_feature_for_vertex.permute(0, 2, 3, 1).reshape((-1, 2 * self.input_channels))
        ).reshape((B, H, W, self.input_channels // 2)).permute(0, 3, 1, 2)
        # output = self.final_aggregation_layer(update_vertex)
        cat_feature_for_edge = torch.cat(
            (
                torch.stack((vertex, vertex, vertex, vertex), dim=-1),
                edge
            ), dim=1
        ).permute(0, 2, 3, 4, 1).reshape((-1, 2 * self.input_channels))
        update_edge = self.edge_update_func(cat_feature_for_edge).reshape((B, H, W, 4, C//2)).permute(0, 4, 1, 2, 3).reshape((-1, 4))
        update_edge_converted = self.update_edge_reduce_func(update_edge).reshape((B, C//2, H, W))

        update_feature = update_vertex * update_edge_converted
        output = self.final_aggregation_layer(
            torch.cat((x, update_feature), dim=1)
        )

        return output

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SegmantationLoss(nn.Module):
    def __init__(self, class_weights=None):
        super(SegmantationLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
    def __call__(self, output, target):
            return self.loss(output, target)
