'''
Author: Shuailin Chen
Created Date: 2021-10-08
Last Modified: 2021-10-08
	content: miscellaneous loading functions
'''


import os.path as osp
import mmcv
import mylib.file_utils as fu
import re

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadSN6Orientation():
    ''' Load orientations of SN6 standard file  '''

    def __init__(self, file_path):
        self.file_path = file_path
        orients = fu.read_file_as_list(file_path)
        self.orients = {l.split()[0]: l.split()[1] for l in orients}

    def __call__(self, results):
        timestamp = re.findall(r'2019\d{10}_2019\d{10}', results['filename'])
        assert len(timestamp) == 1, \
                f'expect #timestamp=1, got {len(timestamp)}'
        timestamp = timestamp[0]
        results['orientation'] = int(self.orients[timestamp])
        return results