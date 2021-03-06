'''
Author: Shuailin Chen
Created Date: 2021-09-14
Last Modified: 2021-09-29
	content: 
'''

_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py',
    '../_base_/datasets/sn6_sar_pro.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
model = dict(
    pretrained='/home/csl/code/PolSAR_SelfSup/work_dirs/pbyol_r50_sn6_sar_pro_ul_ep200/20210929_102939/epoch_200.pth',
    
    decode_head=dict(num_classes=2), auxiliary_head=dict(num_classes=2))
