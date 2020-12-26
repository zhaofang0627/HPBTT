"""
Script for get preprocessed data for deepfashion.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags

import os
import os.path as osp
import numpy as np
import pickle
import cv2

from ..external.hmr.hmr import HMR

flags.DEFINE_string('dataset', 'deepfashion')
flags.DEFINE_integer('img_size', 256, 'image size')

opts = flags.FLAGS

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')

class ShapeTester(object):
    def __init__(self):
        self.hmr = HMR(1)

    def test_and_save(self):
        data_dir = './dataset/DeepFashion/In-shop_Clothes_Retrieval_Benchmark'
        out_dir = './HPBTT/cachedir/deepfashion/data'
        data_cache_dir = './HPBTT/cachedir/deepfashion'

        with open(osp.join(data_dir, 'anno', 'list_landmarks_inshop.txt'), 'r') as f:
            lines = f.readlines()
        line_num = len(lines)
        print(line_num)
        img_dict = pickle.load(open(osp.join(data_cache_dir, 'data', 'train_test_list.pkl'), 'rb'))
        img_list = img_dict['img_list']
        num_imgs = len(img_list)
        print('%d images' % num_imgs)

        theta_num = 0
        for i in range(line_num):
            if i%100 == 0:
                print(i)
            img_info = lines[i].split(' ')
            img_info = list(filter(('').__ne__, img_info))
            img_name = img_info[0]
            if img_name in img_list:
                theta_num += 1
                img_ori = cv2.imread(osp.join(data_dir, img_name))

                scaled_size = round(1.0 * 256)

                vert_shifted, theta, cam_for_render, img_crop = self.hmr.predict(img_ori)

                s = img_name.split('/')[1]
                if not osp.exists(osp.join(out_dir, 'theta', s)):
                    os.mkdir(osp.join(out_dir, 'theta', s))

                g = img_name.split('/')[2]
                if not osp.exists(osp.join(out_dir, 'theta', s, g)):
                    os.mkdir(osp.join(out_dir, 'theta', s, g))

                id = img_name.split('/')[3]
                if not osp.exists(osp.join(out_dir, 'theta', s, g, id)):
                    os.mkdir(osp.join(out_dir, 'theta', s, g, id))

                file_name = img_name.split('/')[4]

                with open(osp.join(out_dir, 'theta', s, g, id, file_name.split('.')[0]+'.pkl'), 'wb') as f:
                    pickle.dump({'theta': theta, 'scaled_size': scaled_size}, f)


def main(_):
    tester = ShapeTester()
    tester.test_and_save()

if __name__ == '__main__':
    app.run(main)
