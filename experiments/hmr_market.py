"""
Script for get preprocessed data for market1501.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags

import os.path as osp
import scipy.io as sio
import cv2
import pickle
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import imsave

from ..external.hmr.hmr import HMR

flags.DEFINE_string('dataset', 'market1501')
flags.DEFINE_integer('img_size', 256, 'image size')

opts = flags.FLAGS

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')

class ShapeTester(object):
    def __init__(self):
        self.hmr = HMR(1)

    def test_and_save(self):
        data_dir = './dataset/market1501'
        out_dir = './HPBTT/cachedir/market1501/data'
        img_dir = osp.join(data_dir, 'images')
        data_cache_dir = osp.join(cache_path, 'market1501')
        anno_path = osp.join(data_cache_dir, 'data', '%s_market1501.mat' % 'train')
        anno = sio.loadmat(anno_path, struct_as_record=False, squeeze_me=True)['images_market']
        num_imgs = len(anno)
        print('%d images' % num_imgs)

        for i in range(num_imgs):
            if i%100 == 0:
                print(i)
            data = anno[i]
            img_path = osp.join(img_dir, str(data.rel_path))
            img_ori = cv2.imread(img_path)

            vert_shifted, theta, cam_for_render, img_crop = self.hmr.predict(img_ori)

            imsave(osp.join(out_dir, 'img_crop', str(data.rel_path).split('.')[0]+'.png'), img_crop)

            with open(osp.join(out_dir, 'theta', str(data.rel_path).split('.')[0]+'.pkl'), 'wb') as f:
                pickle.dump(theta, f)


def main(_):
    tester = ShapeTester()
    tester.test_and_save()

if __name__ == '__main__':
    app.run(main)
