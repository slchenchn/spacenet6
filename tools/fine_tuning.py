'''
Author: Shuailin Chen
Created Date: 2021-10-03
Last Modified: 2021-10-06
	content: fine tuning the self-supervised model
'''

from os import system
import os.path as osp
import subprocess
import argparse
from glob import glob
import re
import natsort
from linear_evaluate import linear_evaluation



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='linear evaluation')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('pth_dir', help='dir of .pth files')
    args = parser.parse_args()
    exclude_keys = ('mmseg_epoch_110.pth', 'mmseg_epoch_120.pth', 'mmseg_epoch_130.pth', 'mmseg_epoch_140.pth', 'mmseg_epoch_150.pth', 'mmseg_epoch_160.pth', 'mmseg_epoch_170.pth', 'mmseg_epoch_180.pth', 'mmseg_epoch_190.pth', 'mmseg_epoch_200.pth')
    exclude_keys = None

    linear_evaluation(args.config, args.pth_dir, exclude_keys=exclude_keys)