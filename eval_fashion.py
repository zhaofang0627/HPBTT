"""
Evaluation for deepfashion.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags, app
import os
import os.path as osp
import numpy as np
import cv2
import pickle
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import imsave


curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, 'cachedir')

flags.DEFINE_string('name', 'exp_name', 'Experiment Name')
flags.DEFINE_string('cache_dir', cache_path, 'Cachedir')
# Set it as split in dataloader
flags.DEFINE_integer('gpu_id', 0, 'Which gpu to use')

flags.DEFINE_integer('batch_size', 4, 'Size of minibatches')
flags.DEFINE_integer('num_train_epoch', 0, 'Number of training iterations')

# Flags for logging and snapshotting
flags.DEFINE_string('checkpoint_dir',
                    osp.join(cache_path, 'snapshots'),
                    'Directory where networks are saved')

flags.DEFINE_string('results_dir_base',
                    osp.join(cache_path, 'evaluation'),
                    'Directory where evaluation results will be saved')

flags.DEFINE_string('results_dir', '', 'This gets set automatically now')

flags.DEFINE_integer('max_eval_iter', 0,
                     'Maximum evaluation iterations. 0 => 1 epoch.')

flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_integer('img_size', 256, 'image size the network was trained on.')
flags.DEFINE_boolean('hmr', True, 'if true do hmr.')

opts = flags.FLAGS


def partition_list(l, partition_size):
    divup = lambda a,b: int((a + b - 1) / b)
    return [l[i*partition_size:(i+1)*partition_size] for i in range(divup(len(l), partition_size))]


def main(_):
    if opts.hmr:
        img_data = pickle.load(open(osp.join('./HPBTT/cachedir/deepfashion/data', 'train_test_list.pkl'), 'rb'))
        query_list = img_data['query_list']
        query_num = len(query_list)
        print(query_num)

        img_list = []
        for i in range(query_num):
            img_rel_path = query_list[i]
            l = img_rel_path.split('/')
            rel_path = osp.join(l[1], l[2], l[3], l[4])
            img_list.append(rel_path)

        img_batch_list = partition_list(img_list, opts.batch_size)
        print(len(img_list))
        print(len(img_batch_list))

        for i in range(len(img_batch_list)-1):
            print(i)
            b = img_batch_list[i]

            img_crop_list = []
            theta_list = []
            for j in range(len(b)):
                img_crop_path = osp.join('/raid/fangzhao/dataset/DeepFashion/In-shop_Clothes_Retrieval_Benchmark', 'img', b[j].split('.')[0] + '.jpg')
                img_crop = cv2.imread(img_crop_path)
                img_crop = img_crop[..., ::-1]
                img_crop_list.append(np.expand_dims(img_crop, 0))

                theta_dict = pickle.load(open(osp.join('./HPBTT/cachedir/deepfashion/data', 'theta', b[j].split('.')[0] + '.pkl'), 'rb'))
                theta_list.append(np.expand_dims(theta_dict['theta'], 0))

            img_crop_batch = np.concatenate(img_crop_list, 0)
            theta_batch = np.concatenate(theta_list, 0)

            batch = {'theta': theta_batch,
                     'img_crop': img_crop_batch,
                     'img_info': b}

            with open('/raid/fangzhao/HPBTT/cachedir/deepfashion/data/batch_hmr_q/batch_hmr_%d.pkl' % i, 'wb') as f:
                pickle.dump(batch, f)

    else:
        from .nnutils import predictor_fashion as pred_util

        predictor = pred_util.MeshPredictor(opts)

        batch_root = '/raid/fangzhao/HPBTT/cachedir/deepfashion/data/batch_hmr_q'

        if not os.path.exists(opts.img_path):
            os.mkdir(opts.img_path)

        batch_path = os.listdir(batch_root)
        for i in range(len(batch_path)):
            print(i)
            batch = pickle.load(open(osp.join(batch_root, batch_path[i]), 'rb'))

            texture_pred_list = []
            for k in range(batch['theta'].shape[0]//opts.batch_size):
                sub_batch = {'theta': batch['theta'][k*opts.batch_size:(k+1)*opts.batch_size],
                             'img_crop': batch['img_crop'][k*opts.batch_size:(k+1)*opts.batch_size]}

                outputs = predictor.predict(sub_batch)
                texture_pred_list.append(outputs['texture_pred'].cpu().numpy())

            texture_pred = np.concatenate(texture_pred_list, 0)

            for ii in range(batch['img_crop'].shape[0]):
                img_name = batch['img_info'][ii]

                s = img_name.split('/')[0]
                if not osp.exists(osp.join(opts.img_path, s)):
                    os.mkdir(osp.join(opts.img_path, s))

                g = img_name.split('/')[1]
                if not osp.exists(osp.join(opts.img_path, s, g)):
                    os.mkdir(osp.join(opts.img_path, s, g))

                id = img_name.split('/')[2]
                if not osp.exists(osp.join(opts.img_path, s, g, id)):
                    os.mkdir(osp.join(opts.img_path, s, g, id))

                file_name = img_name.split('/')[3]

                imsave(osp.join(opts.img_path, s, g, id, file_name.split('.')[0]+'.png'),
                       (np.clip(texture_pred[ii].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8))


if __name__ == '__main__':
    opts.batch_size = 32
    app.run(main)
