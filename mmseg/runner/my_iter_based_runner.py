'''
Author: Shuailin Chen
Created Date: 2021-11-15
Last Modified: 2021-11-15
	content: modification of mmcv iter_based_runner
'''

import torch
from mmcv.runner import IterBasedRunner, EpochBasedRunner, RUNNERS
import time


@RUNNERS.register_module()
class MyIterBasedRunner(IterBasedRunner):

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        ''' The original version makes the val and train step similar, they both process a batch at one step. But the val step should walk through the entire val set, not just one batch.
            This version of val step walks through the entire val set, and discards hooks, inner_iters.
        '''

        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        
        # Adapted from EpochBasedRunner
        # Prevent possible deadlock during epoch transition
        time.sleep(2)  

        for i in range(len(self.data_loader)):
        # for i, data_batch in enumerate(self.data_loader):
            # print(f'i: {i}')
            data_batch = next(data_loader)
            outputs = self.model.val_step(data_batch, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('model.val_step() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
            self.outputs = outputs
            # self._inner_iter += 1

    # def run(self, data_loaders, workflow, max_iters=None, **kwargs):
    #     ''' double the batchsize of val dataloader '''
    #     for wf, dl in zip(workflow, data_loaders):
    #         if 'val' in wf:
    #             dl.
    #     return super().run(data_loaders, workflow, max_iters=max_iters, **kwargs)