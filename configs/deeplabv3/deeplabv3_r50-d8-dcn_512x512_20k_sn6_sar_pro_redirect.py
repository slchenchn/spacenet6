'''
Author: Shuailin Chen
Created Date: 2021-09-14
Last Modified: 2021-10-13
	content: enable deformable convolutions
'''

_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py',
    '../_base_/datasets/sn6_sar_pro_redict.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2)
)


