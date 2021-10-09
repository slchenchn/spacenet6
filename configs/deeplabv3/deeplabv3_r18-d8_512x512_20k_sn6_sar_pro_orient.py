'''
Author: Shuailin Chen
Created Date: 2021-09-14
Last Modified: 2021-10-08
	content: with orientations
'''
_base_ = [
    './deeplabv3_r50-d8_512x512_20k_sn6_sar_pro_orient.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=256,
            in_index=2,
            channels=64,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=2,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='ClassificationHead',
            num_linears=1,
            in_channels=128,
            in_index=1,
            channels=128,
            num_convs=0,
            dropout_ratio=0.1,
            num_classes=2,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
        )])
