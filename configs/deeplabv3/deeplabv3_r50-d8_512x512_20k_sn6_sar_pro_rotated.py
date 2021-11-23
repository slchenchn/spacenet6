'''
Author: Shuailin Chen
Created Date: 2021-09-14
Last Modified: 2021-11-23
	content: 
'''
_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py',
    '../_base_/datasets/sn6_sar_pro_rotated.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
model = dict(
    decode_head=dict(num_classes=2), auxiliary_head=dict(num_classes=2))


# workflow = [('train', 1)]
# evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)