"""
Demo of HMR.

Note that HMR requires the bounding box of the person in the image. The best performance is obtained when max length of the person in the image is roughly 150px. 

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out the bbox and the right scale factor.

Sample usage:

# On images on a tightly cropped image around the person
python -m demo --img_path data/im1963.jpg
python -m demo --img_path data/coco1.png

# On images, with openpose output
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from .src.RunModel import RunModel
from .src import config as hmr_config
from .src.util import renderer as vis_util
from .src.util import image as img_util

class HMR:
    def __init__(self, batch_size=1):
        config = hmr_config.get_config()
        # Using pre-trained model, change this to use your own.
        config.load_path = config.PRETRAINED_MODEL

        config.batch_size_hmr = batch_size
        sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
        self.model = RunModel(config, sess)
        self.config = config

    def preprocess_image(self, img, config):
        img = img[..., ::-1]
        if img.shape[2] == 4:
            img = img[:, :, :3]

        if np.max(img.shape[:2]) != config.img_size_hmr:
            # print('Resizing so the max image size is %d..' % config.img_size_hmr)
            scale = (float(config.img_size_hmr) / np.max(img.shape[:2]))
            scale_cmr = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
            scale_cmr = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]

        crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                                   config.img_size_hmr)

        crop_cmr, proc_param_cmr = img_util.scale_and_crop(img, scale_cmr, center,
                                                   config.img_size)

        # cv2.imwrite("hmr_input" + ".jpg", crop[..., ::-1])
        # Normalize image to [-1, 1]
        crop = 2 * ((crop / 255.) - 0.5)

        return crop, proc_param, img, crop_cmr

    def visualize(self, img, proc_param, joints, verts, cam):
        """
        Renders the result in original image coordinate frame.
        """
        cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
            proc_param, verts, cam, joints, img_size=img.shape[:2])
        return cam_for_render, vert_shifted

    def predict(self, img_bgr):
        crop, proc_param, img, crop_cmr = self.preprocess_image(img_bgr, self.config)
        # print(crop.shape)
        # Add batch dimension: 1 x D x D x 3
        input_img = np.expand_dims(crop, 0)

        joints, verts, cams, joints3d, theta = self.model.predict(
            input_img, get_theta=True)

        cam_for_render, vert_shifted = self.visualize(img, proc_param, joints[0], verts[0], cams[0])

        return vert_shifted, theta[0], cam_for_render, crop_cmr

    def predict_batch(self, img_bgr_batch):
        crop_list = []
        proc_param_list = []
        img_list = []
        crop_cmr_list = []
        for i in range(img_bgr_batch.shape[0]):
            crop, proc_param, img, crop_cmr = self.preprocess_image(img_bgr_batch[i], self.config)
            # print(crop.shape)
            crop_list.append(np.expand_dims(crop, 0))
            proc_param_list.append(proc_param)
            img_list.append(img)
            crop_cmr_list.append(np.expand_dims(crop_cmr, 0))

        # Batch dimension: N x D x D x 3
        input_img_batch = np.concatenate(crop_list, 0)
        crop_cmr_batch = np.concatenate(crop_cmr_list, 0)

        joints, verts, cams, joints3d, theta = self.model.predict(
            input_img_batch, get_theta=True)

        return theta, crop_cmr_batch


