'''
Author: Shuailin Chen
Created Date: 2021-10-03
Last Modified: 2021-10-06
	content: linear evaluation of self-supervised model
'''

from os import system
import os.path as osp
import subprocess
import argparse
from glob import glob
import re
import natsort


def linear_evaluation(config, pth_dir, exclude_keys=None):
    pths = glob(osp.join(pth_dir, 'mmseg*.pth'))
    pths = natsort.natsorted(pths)
    print(f'totally {len(pths)} pth files')
    for pth in pths:
        if (exclude_keys is not None) and (osp.basename(pth) in exclude_keys):
            continue

        work_dir = re.sub(r'^.*?/work_dirs/', r'work_dirs/', pth)
        work_dir = osp.splitext(work_dir)[0]
        cmd = f'python tools/train.py {config} --work-dir {work_dir} --options model.pretrained={pth}'
        print(f'executing {cmd}')
        system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='linear evaluation')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('pth_dir', help='dir of .pth files')
    args = parser.parse_args()
    # exclude_keys = ('mmseg_epoch_90.pth', 'mmseg_epoch_190.pth')
    exclude_keys = None

    linear_evaluation(args.config, args.pth_dir, exclude_keys=exclude_keys)