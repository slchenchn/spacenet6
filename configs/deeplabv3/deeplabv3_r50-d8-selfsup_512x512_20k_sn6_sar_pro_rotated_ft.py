'''
Author: Shuailin Chen
Created Date: 2021-09-14
Last Modified: 2021-10-25
	content: fine tuning
'''

_base_ = [
    './deeplabv3_r50-d8_512x512_20k_sn6_sar_pro_rotated.py'
]

model = dict(
    pretrained='/home/csl/code/PolSAR_SelfSup/work_dirs/pbyol_r18_sn6_sar_pro_ul_ep400_lr03/20211009_142911/mmseg_epoch_400.pth',
)
