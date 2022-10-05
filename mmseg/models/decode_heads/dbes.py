import logging
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmseg.core import add_prefix
from mmseg.datasets.pipelines import RelaxedBoundaryLossToTensor
from mmseg.ops import resize
from ..builder import HEADS, build_loss
from ..losses import accuracy, onehot2label
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class FCNDBES(BaseDecodeHead):
    """GFPN with DBES
    """
    def __init__(self, **kargs):
        super().__init__(**kargs)
        # delattr(self, 'conv_seg')     # 这个不能删，因为init_cfg还是包含了这个模块，删了会出错

        c1 = sum(self.in_channels)
        c2 = self.in_channels[1]
        self.pre_conv = nn.Conv2d(c1, 256, 1, bias=False)

        Norm2d = partial(build_norm_layer, self.norm_cfg)
        self.squeeze_body_edge = SqueezeBodyEdge(256, Norm2d)
        self.bot_fine = nn.Conv2d(c2, 48, kernel_size=1, bias=False)
        self.edge_fusion = nn.Conv2d(256 + 48, 256, 1, bias=False)
        self.edge_out = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=3, padding=1, bias=False),
            Norm2d(48)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 1, kernel_size=1, bias=False)
        )

        self.final_seg = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_classes, kernel_size=1, bias=False))

        self.dsn_seg_body = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_classes, kernel_size=1, bias=False)
        )
        self.sigmoid_edge = nn.Sigmoid()

    def cls_seg(self, feat):
        raise NotImplementedError(f'this is implented in forward()')

    def forward(self, inputs):

        inputs = self._transform_inputs(inputs)
        x1 = inputs[1]      # c2 feature
        x = torch.cat([resize(
                input=x,
                size=inputs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for x in inputs
            ], dim=1)
        x = self.pre_conv(x)    # change dim from 270 to 256
        seg_body, seg_edge = self.squeeze_body_edge(x)

        fine_size = x1.size()
        dec0_fine = self.bot_fine(x1)

        seg_edge = self.edge_fusion(torch.cat([Upsample(seg_edge, fine_size[2:]), dec0_fine], dim=1))
        seg_edge_out = self.edge_out(seg_edge)

        seg_out = seg_edge + Upsample(seg_body, fine_size[2:])
        x = Upsample(x, fine_size[2:])

        seg_out = torch.cat([x, seg_out], dim=1)
        seg_final_out = self.final_seg(seg_out)

        seg_edge_out = self.sigmoid_edge(seg_edge_out)
        seg_body_out = self.dsn_seg_body(seg_body)

        if seg_final_out.isnan().any():
            print('breakpoint')
        return seg_final_out, seg_body_out, seg_edge_out

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label, **kargs):
        """Compute segmentation loss."""
        loss = dict()
        seg_label = seg_label.squeeze()
        seg_logit = [resize(
            input=logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners) for logit in seg_logit]
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index,
                    **kargs)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index,
                    **kargs)

        loss['acc_seg'] = accuracy(
            seg_logit[0], onehot2label(seg_label), ignore_index=self.ignore_index)

        for k, v in loss.copy().items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    loss[f'{k}_{k2}'] = v2
                del loss[k]
        return loss

    def forward_test(self, inputs, img_metas, test_cfg):
        final, body, edge = self.forward(inputs)
        return final


@HEADS.register_module()
class PSFCNDBES(FCNDBES):
    def __init__(self, ps_thres=0.8, num_classes=2, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        self.ps_thres = ps_thres
        self.prob_thres = math.exp(-ps_thres)
        self.proc_ps = RelaxedBoundaryLossToTensor(
            num_classes=num_classes,
            ignore_id=self.ignore_index)

    def _extract_feat(self, labeled, unlabeled, label_size):

        # split two temporals
        unlabel1 = unlabeled["feat"][0]
        unlabel2 = unlabeled["feat"][1]

        logits_labeled = self.forward(labeled["feat"])
        logits_unlabel1 = self.forward(unlabel1)
        logits_unlabel2 = self.forward(unlabel2)

        logits_unlabel1 = [resize(
            input=logit,
            size=label_size,
            mode="bilinear",
            align_corners=self.align_corners,
        )for logit in logits_unlabel1]

        logits_unlabel2 = [resize(
            input=logit,
            size=label_size,
            mode="bilinear",
            align_corners=self.align_corners,
        ) for logit in logits_unlabel2]

        val1, ps1 = logits_unlabel1[0].max(dim=1, keepdims=True)
        val2, ps2 = logits_unlabel2[0].max(dim=1, keepdims=True)
        ps1[val1 < self.prob_thres] = self.ignore_index
        ps2[val2 < self.prob_thres] = self.ignore_index
        return logits_labeled, logits_unlabel1, logits_unlabel2, ps1, ps2

    def forward_train(self, labeled, unlabeled, train_cfg, **kargs):
        logits_labeled, logits_unlabel, logits_unlabe2, ps1, ps2 = self._extract_feat(
            labeled, unlabeled, labeled["gt_semantic_seg"].shape[-2:]
        )

        invalids = unlabeled["gt_semantic_seg"].squeeze().bool()
        invalids1 = invalids[..., 0].unsqueeze(1)
        invalids2 = invalids[..., 1].unsqueeze(1)

        ps1[invalids1] = self.ignore_index
        ps2[invalids2] = self.ignore_index

        batchsize = ps1.shape[0]
        ps1 = ps1.cpu().numpy().squeeze()
        ps2 = ps2.cpu().numpy().squeeze()
        new_ps1 = [0] * batchsize
        new_ps2 = [0] * batchsize
        for i in range(batchsize):
            cur_ps1 = ps1[i]
            cur_ps2 = ps2[i]
            cur_ps1 = self.proc_ps(dict(gt_semantic_seg=cur_ps1))['gt_semantic_seg']
            cur_ps2 = self.proc_ps(dict(gt_semantic_seg=cur_ps2))['gt_semantic_seg']
            new_ps1[i] = cur_ps1
            new_ps2[i] = cur_ps2
        ps1 = np.stack(new_ps1, axis=0)
        ps2 = np.stack(new_ps2, axis=0)
        ps1 = torch.from_numpy(ps1).to(invalids.device)
        ps2 = torch.from_numpy(ps2).to(invalids.device)

        loss_labeled = self.losses(logits_labeled, labeled["gt_semantic_seg"])
        loss_unlabele1 = self.losses(logits_unlabel, ps1)
        loss_unlabele2 = self.losses(logits_unlabe2, ps2)

        losses = dict()
        losses.update(add_prefix(loss_labeled, "labeled"))
        losses.update(add_prefix(loss_unlabele1, "unlabel1"))
        losses.update(add_prefix(loss_unlabele2, "unlabel2"))
        return losses


@HEADS.register_module()
class PSFCNDBESV2(PSFCNDBES):

    def forward_train(self, labeled, unlabeled, train_cfg, **kargs):
        logits_labeled, logits_unlabel, logits_unlabe2, ps1, ps2 = self._extract_feat(
            labeled, unlabeled, labeled["gt_semantic_seg"].shape[-2:]
        )

        invalids = unlabeled["gt_semantic_seg"].squeeze().bool()
        invalids1 = invalids[..., 0].unsqueeze(1)
        invalids2 = invalids[..., 1].unsqueeze(1)

        ps1[invalids1] = self.ignore_index
        ps2[invalids2] = self.ignore_index

        batchsize = ps1.shape[0]
        ps1 = ps1.cpu().numpy().squeeze()
        ps2 = ps2.cpu().numpy().squeeze()
        new_ps1 = [0] * batchsize
        new_ps2 = [0] * batchsize
        for i in range(batchsize):
            cur_ps1 = ps1[i]
            cur_ps2 = ps2[i]
            cur_ps1 = self.proc_ps(dict(gt_semantic_seg=cur_ps1))['gt_semantic_seg']
            cur_ps2 = self.proc_ps(dict(gt_semantic_seg=cur_ps2))['gt_semantic_seg']
            new_ps1[i] = cur_ps1
            new_ps2[i] = cur_ps2
        ps1 = np.stack(new_ps1, axis=0)
        ps2 = np.stack(new_ps2, axis=0)
        ps1 = torch.from_numpy(ps1).to(invalids.device)
        ps2 = torch.from_numpy(ps2).to(invalids.device)

        loss_labeled = self.losses(logits_labeled, labeled["gt_semantic_seg"])
        loss_unlabele1 = self.losses(logits_unlabel, ps1, edge_weight=0.1)
        loss_unlabele2 = self.losses(logits_unlabe2, ps2, edge_weight=0.1)

        losses = dict()
        losses.update(add_prefix(loss_labeled, "labeled"))
        losses.update(add_prefix(loss_unlabele1, "unlabel1"))
        losses.update(add_prefix(loss_unlabele2, "unlabel2"))
        return losses


@HEADS.register_module()
class PSFCNDBESV3(PSFCNDBES):
    def __init__(self, unsup_loss, **kwargs):
        super().__init__(**kwargs)
        self.unsup_loss = build_loss(unsup_loss)

    def forward_train(self, labeled, unlabeled, train_cfg, **kargs):
        logits_labeled, logits_unlabel, logits_unlabe2, ps1, ps2 = self._extract_feat(
            labeled, unlabeled, labeled["gt_semantic_seg"].shape[-2:]
        )

        invalids = unlabeled["gt_semantic_seg"].squeeze().bool()
        invalids1 = invalids[..., 0].unsqueeze(1)
        invalids2 = invalids[..., 1].unsqueeze(1)

        ps1[invalids1] = self.ignore_index
        ps2[invalids2] = self.ignore_index

        loss_labeled = self.losses(logits_labeled, labeled["gt_semantic_seg"])
        loss_unlabele1 = self.unsup_lossess(logits_unlabel[0], ps1)
        loss_unlabele2 = self.unsup_lossess(logits_unlabe2[0], ps2)

        losses = dict()
        losses.update(add_prefix(loss_labeled, "labeled"))
        losses.update(add_prefix(loss_unlabele1, "unlabel1"))
        losses.update(add_prefix(loss_unlabele2, "unlabel2"))
        return losses

    @force_fp32(apply_to=('seg_logit', ))
    def unsup_lossess(self, seg_logit, seg_label, **kargs):
        """Compute segmentation loss."""
        loss = dict()
        # seg_label = seg_label.squeeze()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[-2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.unsup_loss]
        else:
            losses_decode = self.unsup_loss
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index,
                    **kargs)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index,
                    **kargs)

        loss['acc_seg'] = accuracy(
            seg_logit, seg_label, ignore_index=self.ignore_index)

        for k, v in loss.copy().items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    loss[f'{k}_{k2}'] = v2
                del loss[k]
        return loss


class SqueezeBodyEdge(nn.Module):
    def __init__(self, inplane, norm_layer):
        """
        implementation of body generation part
        :param inplane:
        :param norm_layer:
        """
        super(SqueezeBodyEdge, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            norm_layer(inplane)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            norm_layer(inplane)[1],
            nn.ReLU(inplace=True)
        )

        self.flow_make = nn.Conv2d(inplane *2 , 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        size = x.size()[2:]
        seg_down = self.down(x)
        seg_down = F.upsample(seg_down, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([x, seg_down], dim=1))
        seg_flow_warp = self.flow_warp(x, flow, size)
        seg_edge = x - seg_flow_warp
        return seg_flow_warp, seg_edge

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)

        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output


def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return F.interpolate(x, size=size, mode='bilinear',
                                    align_corners=True)


                                    