"""
Base data loading class.

Should output:
    - img: B X 3 X H X W
    - mask: B X H X W
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from absl import flags, app
import pickle
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from ..utils import image as image_utils
from .sampler import RandomIdentitySampler


flags.DEFINE_float('padding_frac', 0.05,
                   'bbox is increased by this fraction of max_dim')

flags.DEFINE_float('jitter_frac', 0.05,
                   'bbox is jittered by this fraction of max_dim')

flags.DEFINE_enum('split', 'train', ['train', 'val', 'all', 'test'], 'eval split')
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
        # Child class should define/load:
        self.opts = opts
        self.img_size = opts.img_size
        self.filter_key = filter_key
    
    def forward_img(self, index):
        data = self.anno[index]

        img_path = osp.join(self.img_dir, str(data.rel_path).split('.')[0]+'.png')
        img = np.array(pil_loader(img_path)) / 255.0

        mask = np.expand_dims(data.mask, 2)

        theta = pickle.load(open(osp.join(self.theta_dir, str(data.rel_path).split('.')[0]+'.pkl'), 'rb'), encoding='latin1')

        # Adjust to 0 indexing
        bbox = np.array(
            [data.bbox.x1, data.bbox.y1, data.bbox.x2, data.bbox.y2], float) - 1

        parts = data.parts.T.astype(float)
        kp = np.copy(parts)
        vis = kp[:, 2] > 0
        kp[vis, :2] -= 1

        bbox = image_utils.square_bbox(bbox)
        mask = image_utils.crop(mask, bbox, bgval=0)
        mask, _ = image_utils.resize_img(mask, 2.0)

        # Finally transpose the image to 3xHxW
        img = np.transpose(img, (2, 0, 1))

        return img, mask, theta

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, index):
        img, mask, theta = self.forward_img(index)

        elem = {
            'img': img,
            'mask': mask,
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

