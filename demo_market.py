"""
Demo of Market.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags, app
import os.path as osp
import numpy as np
import skimage.io as io
import cv2
import pickle
import matplotlib
matplotlib.use('Agg')

from .utils import image as img_util

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
flags.DEFINE_boolean('test', True, 'if true test.')
flags.DEFINE_boolean('hmr', True, 'if true do hmr.')

opts = flags.FLAGS


def preprocess_image(img_path, img_size=256):
    img = io.imread(img_path) / 255.

    # Scale the max image size to be img_size
    scale_factor = float(img_size) / np.max(img.shape[:2])
    img, _ = img_util.resize_img(img, scale_factor)

    # Crop img_size x img_size from the center
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)
    # img center in (x, y)
    center = center[::-1]
    bbox = np.hstack([center - img_size / 2., center + img_size / 2.])

    img = img_util.crop(img, bbox, bgval=1.)

    # Transpose the image to 3xHxW
    img = np.transpose(img, (2, 0, 1))

    return img


def visualize(img, outputs, renderer):
    vert = outputs['verts_stand'][0]
    texture = outputs['texture'][0]
    shape_pred = renderer(vert, cams=None)
    img_pred = renderer(vert, cams=None, texture=texture)

    # Different viewpoints.
    vp1 = renderer.rotated(vert, deg=120, axis=[0, 1, 0], cam=None, texture=texture)
    vp2 = renderer.rotated(vert, deg=240, axis=[0, 1, 0], cam=None, texture=texture)
    vp3 = renderer.rotated(vert, deg=90, axis=[1, 0, 0], cam=None, texture=texture)

    # img = np.transpose(img, (1, 2, 0))
    img = img[..., ::-1]
    import matplotlib.pyplot as plt
    # plt.ion()
    plt.figure(1)
    plt.clf()
    plt.subplot(231)
    plt.imshow(img)
    plt.title('input')
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(shape_pred)
    plt.title('pred mesh')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(img_pred)
    plt.title('pred mesh w/texture')
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(vp1)
    plt.title('different viewpoints')
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(vp2)
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(vp3)
    plt.axis('off')
    plt.draw()
    #plt.show()
    plt.savefig("./HPBTT/demo_data/show.png")
    # import ipdb
    # ipdb.set_trace()


def main(_):
    img = cv2.imread(opts.img_path)

    if opts.hmr:
        from .external.hmr.hmr import HMR

        hmr = HMR(1)
        vert_shifted, theta, cam_for_render, img_crop = hmr.predict(img)

        batch = {'theta': np.expand_dims(theta, 0),
                 'img_crop': np.expand_dims(img_crop, 0),
                 }

        with open('./HPBTT/batch_hmr.pkl', 'wb') as f:
            pickle.dump(batch, f)
    else:
        from .nnutils import predictor_market as pred_util
        batch = pickle.load(open('./HPBTT/batch_hmr.pkl', 'rb'))

        predictor = pred_util.MeshPredictor(opts)
        outputs = predictor.predict(batch)

        # This is resolution
        renderer = predictor.vis_rend
        renderer.set_light_dir([0, 1, -1], 0.4)

        visualize(img, outputs, predictor.vis_rend)


if __name__ == '__main__':
    opts.batch_size = 1
    app.run(main)
