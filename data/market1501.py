from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import pickle
import scipy.io as sio
from absl import flags, app

from . import base as base_data

# -------------- flags ------------- #
# ---------------------------------- #
kData = './HPBTT/cachedir/market1501/data'
    
flags.DEFINE_string('data_dir', kData, 'Market1501 Data Directory')
flags.DEFINE_integer('img_size', 256, 'image size')
flags.DEFINE_integer('img_size_hmr', 224, 'image size hmr')

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')
flags.DEFINE_string('data_cache_dir', osp.join(cache_path, 'market1501'), 'Market1501 Data Directory')

opts = flags.FLAGS

# -------------- Dataset ------------- #
# ------------------------------------ #
class Market1501(base_data.BaseDataset):
    '''
    Market1501 Data loader
    '''

    def __init__(self, opts, filter_key=None):
        super(Market1501, self).__init__(opts, filter_key=filter_key)
        self.data_dir = opts.data_dir
        self.data_cache_dir = opts.data_cache_dir

        self.img_dir = osp.join(self.data_dir, 'img_crop')
        self.theta_dir = osp.join(self.data_dir, 'theta')
        self.anno_path = osp.join(self.data_cache_dir, 'data', '%s_market1501.mat' % opts.split)

        if not osp.exists(self.anno_path):
            print('%s doesnt exist!' % self.anno_path)
            import ipdb; ipdb.set_trace()
        self.filter_key = filter_key

        # Load the annotation file.
        print('loading %s' % self.anno_path)
        self.anno = sio.loadmat(
            self.anno_path, struct_as_record=False, squeeze_me=True)['images_market']

        self.num_imgs = len(self.anno)
        print('%d images' % self.num_imgs)
        self.pids_dict = pickle.load(open(osp.join(self.data_dir, 'pids_dict.pkl'), 'rb'), encoding='latin1')

        self.data_list = []
        for i in range(self.num_imgs):
            pid = self.pids_dict[int(self.anno[i].rel_path.split('_')[0])]
            theta = pickle.load(open(osp.join(self.theta_dir, str(self.anno[i].rel_path).split('.')[0]+'.pkl'), 'rb'), encoding='latin1')
            poses = theta[3:(3 + 72)]
            self.data_list.append((i, pid, poses))


#----------- Data Loader ----------#
#----------------------------------#
def data_loader(opts, shuffle=True):
    return base_data.base_loader(Market1501, opts.batch_size, opts, filter_key=None, shuffle=shuffle)


def data_id_loader(opts, shuffle=True):
    return base_data.base_id_loader(Market1501, opts.batch_size, opts, filter_key=None, shuffle=shuffle)
