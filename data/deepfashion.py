from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import pickle
from absl import flags, app

from . import base_fashion as base_data

# -------------- flags ------------- #
# ---------------------------------- #
kData = '/raid/fangzhao/HPBTT/cachedir/deepfashion/data'
    
flags.DEFINE_string('data_dir', kData, 'Deepfashion Data Directory')
flags.DEFINE_integer('img_size', 256, 'image size')
flags.DEFINE_integer('img_size_hmr', 224, 'image size hmr')

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')
flags.DEFINE_string('data_cache_dir', osp.join(cache_path, 'deepfashion'), 'Deepfashion Data Directory')

opts = flags.FLAGS

# -------------- Dataset ------------- #
# ------------------------------------ #
class Deepfashion(base_data.BaseDataset):
    '''
    Market1501 Data loader
    '''

    def __init__(self, opts, filter_key=None):
        super(Deepfashion, self).__init__(opts, filter_key=filter_key)
        self.data_dir = opts.data_dir
        self.data_cache_dir = opts.data_cache_dir

        self.ori_img_dir = osp.join('/raid/fangzhao/dataset/DeepFashion/In-shop_Clothes_Retrieval_Benchmark', 'img')
        self.theta_dir = osp.join(self.data_dir, 'theta')
        self.mask_ori_dir = osp.join(self.data_dir, 'seg')

        self.filter_key = filter_key

        self.train_test_data = pickle.load(open(osp.join(self.data_dir, 'train_test_list.pkl'), 'rb'))

        self.train_list = self.train_test_data['train_list']

        self.num_imgs = len(self.train_list)
        print('%d images' % self.num_imgs)

        self.data_list = []
        for i in range(self.num_imgs):
            l = self.train_list[i].split('/')
            pid = l[1] + '_' + l[2] + '_' + l[3] + '_' + l[4].split('_')[0]
            # print(pid)
            self.data_list.append((i, pid))


#----------- Data Loader ----------#
#----------------------------------#
def data_loader(opts, shuffle=True):
    return base_data.base_loader(Deepfashion, opts.batch_size, opts, filter_key=None, shuffle=shuffle)


def data_id_loader(opts, shuffle=True):
    return base_data.base_id_loader(Deepfashion, opts.batch_size, opts, filter_key=None, shuffle=shuffle)
