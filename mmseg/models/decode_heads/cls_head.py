'''
Author: Shuailin Chen
Created Date: 2021-10-08
Last Modified: 2021-10-08
	content: classification head
'''

import torch
from torch import nn
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from .decode_head import BaseDecodeHead
from .fcn_head import FCNHead
from ..builder import HEADS, build_loss
from ..losses import accuracy


@HEADS.register_module()
class ClassificationHead(FCNHead):
    ''' Classfication head 
    NOTE: compared with BaseDecodeHead, `ignore` and `sampler` is removed

    '''

    def __init__(self, *args, num_linears=1, **kargs):
        super().__init__(*args, sampler=None, concat_input=False, **kargs)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        if num_linears == 0:
            self.linears = nn.Identity()
        else:
            linears = []
            for ii in range(num_linears):
                linears.append(
                    nn.Linear(self.channels, self.channels)
                )
            self.linears = nn.Sequential(*linears)
        
        self.cls = nn.Linear(self.channels, self.num_classes)


    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        cls_logits = self.forward(inputs)
        cls_label = [ii['orientation'] for ii in img_metas]
        cls_label = torch.tensor(cls_label, device=inputs[0].device)
        losses = self.losses(cls_logits, cls_label)
        return losses

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        x = self.convs(x)
        x = self.pool(x).squeeze()
        assert x.ndim==2, f'expect ndim of x=2, got {x.ndim}'
        x = self.linears(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.cls(x)
        return x
        
    @force_fp32(apply_to=('cls_logits', ))
    def losses(self, cls_logits, cls_label):
        """Compute segmentation loss."""
        loss = dict()
        for loss_decode in self.loss_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    cls_logits,
                    cls_label,
                    weight=None,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    cls_logits,
                    cls_label,
                    weight=None,
                    ignore_index=self.ignore_index)

        loss['acc_cls'] = accuracy(cls_logits, cls_label)
        return loss
