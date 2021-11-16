'''
Author: Shuailin Chen
Created Date: 2021-09-14
Last Modified: 2021-11-16
	content: 
'''
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
# workflow = [('train', 1)]

# number after 'train' mean iters; while numbers after 'val' means epochs
workflow = [('train', 500), ('val', 1)]
# workflow = [('val', 1), ('train', 1)]
cudnn_benchmark = True
