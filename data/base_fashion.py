"""
Base data loading class.

Should output:
    - img: B X 3 X H X W
    - kp: B X nKp X 2
    - mask: B X H X W
    - sfm_pose: B X 7 (s, tr, q)
    (kp, sfm_pose) correspond to image coordinates in [-1, 1]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np

import scipy.misc
import scipy.linalg
import scipy.ndimage.interpolation
from absl import flags, app
from copy import deepcopy
import cv2
import pickle
# from matplotlib.pyplot import imread
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from ..utils import image as image_utils
from ..utils import transformations
from .sampler import RandomIdentitySampler


# flags.DEFINE_integer('img_size', 256, 'image size')

flags.DEFINE_float('padding_frac', 0.05,
                   'bbox is increased by this fraction of max_dim')

flags.DEFINE_float('jitter_frac', 0.05,
                   'bbox is jittered by this fraction of max_dim')

flags.DEFINE_enum('split', 'train', ['train', 'val', 'all', 'test'], 'eval split')
# flags.DEFINE_integer('num_kps', 15, 'The dataloader should override these.')
flags.DEFINE_integer('n_data_workers', 4, 'Number of data loading workers')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


# -------------- Dataset ------------- #
# ------------------------------------ #
class BaseDataset(Dataset):
    ''' 
    img, mask, pose data loader
    '''

    def __init__(self, opts, filter_key=None):
        self.opts = opts
        self.img_size = opts.img_size
        self.filter_key = filter_key
    
    def forward_img(self, index):
        img_rel_path = self.train_list[index]

        l = img_rel_path.split('/')
        rel_path = osp.join(l[1], l[2], l[3], l[4])
        ori_img_path = osp.join(self.ori_img_dir, rel_path.split('.')[0] + '.jpg')

        img_ori_rgb = pil_loader(ori_img_path)
        img_ori = np.array(img_ori_rgb) / 255.0

        theta_dict = pickle.load(open(osp.join(self.theta_dir, rel_path.split('.')[0]+'.pkl'), 'rb'))

        mask_ori_path = osp.join(self.mask_ori_dir, rel_path.split('.')[0] + '.png')
        mask_ori_rgb = pil_loader(mask_ori_path)
        mask_ori = np.array(mask_ori_rgb) / 255.0

        # Finally transpose the image to 3xHxW
        img_ori = np.transpose(img_ori, (2, 0, 1))
        mask_ori = np.mean(mask_ori, 2)

        return theta_dict['theta'], img_ori, mask_ori

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, index):
        theta, img_ori, mask_ori = self.forward_img(index)

        elem = {
            'img_ori': img_ori,
            'mask_ori': mask_ori,
            'theta': theta,
        }

        return elem

# ------------ Data Loader ----------- #
# ------------------------------------ #
def base_loader(d_set_func, batch_size, opts, filter_key=None, shuffle=True):
    dset = d_set_func(opts, filter_key=filter_key)
    return DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=opts.n_data_workers,
        drop_last=True)

def base_id_loader(d_set_func, batch_size, opts, filter_key=None, shuffle=True):
    dset = d_set_func(opts, filter_key=filter_key)
    return DataLoader(
        dset,
        batch_size=batch_size,
        sampler=RandomIdentitySampler(dset.data_list, 2),
        num_workers=opts.n_data_workers,
        drop_last=True)
