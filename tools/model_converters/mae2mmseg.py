'''
Author: Shuailin Chen
Created Date: 2021-12-09
Last Modified: 2021-12-09
	content:  convert ViT implementation of MAE-pytorch (https://github.com/pengzhiliang/MAE-pytorch) to mmseg
'''

import argparse
import os.path as osp
from collections import OrderedDict

import mmcv
import torch
from mmcv.runner import CheckpointLoader

import vit2mmseg


def convert_vit_mae(ckpt):
    ''' Convert the ViT checkpoint generated by MAE-pytorch
    NOTE: the cls token and pos_embed is missing is implementation of MAE-pytorch

    '''

    # extract encoder
    encoder = dict()
    for k, v in ckpt.items():
        if k.startswith('encoder.'):
            new_k = k.replace('encoder.', '')
            encoder.update({new_k: v})
    # print(encoder.keys())

    # use mmseg official implementation
    middle = vit2mmseg.convert_vit(encoder)

    # concate q/v_bias to get in_proj_bias
    new_ckpt = OrderedDict()
    for k, v in middle.items():
        if 'q_bias' in k:
            new_k = k.replace('q_bias', 'attn.in_proj_bias')
            new_v = torch.cat((middle[k], torch.zeros_like(middle[k], requires_grad=False), middle[k.replace('q_bias', 'v_bias')]))
        elif 'v_bias' in k:
            continue
        else:
            new_k = k
            new_v = v

        new_ckpt.update({new_k: new_v})
    
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in timm pretrained vit models to '
        'MMSegmentation style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    if 'state_dict' in checkpoint:
        # timm checkpoint
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        # deit checkpoint
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    weight = convert_vit_mae(state_dict)
    mmcv.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()