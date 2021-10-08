'''
Author: Shuailin Chen
Created Date: 2021-09-14
Last Modified: 2021-10-08
	content: 
'''

norm_cfg = dict(type='SyncBN', requires_grad=True)

_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py',
    '../_base_/datasets/sn6_sar_pro_orient.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
model = dict(
    decode_head=dict(num_classes=2),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=1024,
            in_index=2,
            channels=256,
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
            in_channels=1024,
            in_index=2,
            channels=1024,
            num_convs=0,
            dropout_ratio=0.1,
            num_classes=2,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
        )]
    )
