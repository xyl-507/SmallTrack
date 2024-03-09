# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F
import torch

from siamban.core.config import cfg
from siamban.models.loss import select_cross_entropy_loss, select_iou_loss
from siamban.models.backbone import get_backbone
from siamban.models.head import get_ban_head
from siamban.models.neck import get_neck
from siamban.models.gal.gal import GEM


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        self.gem = GEM(sync_bn=True, input_channels=2)  # xyl 20221003   cls channel:2, loc channel: 4

        # build ban head
        if cfg.BAN.BAN:
            self.head = get_ban_head(cfg.BAN.TYPE,
                                     **cfg.BAN.KWARGS)

    def avg(self, lst):
        return sum(lst) / len(lst)

    def weighted_avg(self, lst, weight):
        s = 0
        for i in range(len(weight)):
            s += lst[i] * weight[i]
        return s

    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        if not cfg.ADJUST.LAYER:
            if cfg.ADJUST.FUSE == 'wavg':
                cls_weight = self.rpn_head.cls_weight
                self.cf = self.weighted_avg([cf for cf in xf], cls_weight)
            elif cfg.ADJUST.FUSE == 'avg':
                self.cf = self.avg([cf for cf in xf])
            elif cfg.ADJUST.FUSE == 'con':
                self.cf = torch.cat([cf for cf in xf], dim=1)
        else:
            if isinstance(xf, list):
                self.cf = xf[cfg.ADJUST.LAYER - 1]
            else:
                self.cf = xf
        cls, loc = self.head(self.zf, xf)
        cls = self.gem(cls)  # xyl 20221003

        if cfg.TRACK.TEMPLATE_UPDATE:

            if cfg.BAN.BAN:
                cls_st, loc_st = self.head(self.zf_st, xf)
            else:
                b, _, h, w = xf.size()
                cls_st = F.conv2d(xf.view(1, -1, h, w), self.zf_st, groups=b).transpose(0, 1)
            return {
                'cls': cls,
                # 'loc': loc if cfg.RPN.RPN else None,
                'loc': loc if cfg.BAN.BAN else None,
                'cls_st': cls_st,
                'loc_st': loc_st if cfg.BAN.BAN else None,
            }
        else:
            return {
                'cls': cls,
                'loc': loc if cfg.BAN.BAN else None,
            }

    def log_softmax(self, cls):
        if cfg.BAN.BAN:
            cls = cls.permute(0, 2, 3, 1).contiguous()
            cls = F.log_softmax(cls, dim=3)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        cls, loc = self.head(zf, xf)
        cls = self.gem(cls)  # xyl 20221003

        # get loss

        # cls loss with cross entropy loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)

        # loc loss with iou loss
        loc_loss = select_iou_loss(loc, label_loc, label_cls)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                                cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        return outputs
