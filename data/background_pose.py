import os

import cv2
import numpy as np
from absl import flags, app
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from .data_utils import RandomCrop
from ..external.hmr.src.util import image as img_util

import tqdm

bgData = '/raid/fangzhao/dataset/PRW-v16.04.20/frames'
flags.DEFINE_string('PRW_img_path', bgData, 'Background Data Directory')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class BackgroundDataset(Dataset):

    def __getitem__(self, index):
        texture_img_path = self.data[index]
        texture_img = np.array(pil_loader(texture_img_path))  # / 255.0
        if texture_img is None or texture_img.shape[0] <= 0 or texture_img.shape[1] <= 0:
            return self.__getitem__(np.random.randint(0, self.__len__()))
        texture_img = self.random_crop(texture_img)
        if np.random.rand(1) > 0.5:
            # Need copy bc torch collate doesnt like neg strides
            texture_img = texture_img[:, ::-1, :]

        texture_img, _ = img_util.scale_and_crop(texture_img, self.scale_cmr, self.center, self.img_size_cmr)

        # Finally transpose the image to 3xHxW
        texture_img = texture_img / 255.0
        texture_img = np.transpose(texture_img, (2, 0, 1))

        return {'bg_img': texture_img}

    def __len__(self):
        return len(self.data)

    def __init__(self, opts, data_path_list, img_size=(128, 64)):
        self.data_path_list = data_path_list
        self.img_size = img_size
        self.img_size_cmr = opts.img_size
        self.scale_cmr = (float(opts.img_size) / max(img_size))
        center = np.round(np.array(img_size) / 2).astype(int)
        # image center in (x,y)
        self.center = center[::-1]
        self.data = []
        self.generate_index()

        self.random_crop = RandomCrop(output_size=self.img_size)

    def generate_index(self):
        print('generating background index')
        for data_path in self.data_path_list:
            for root, dirs, files in os.walk(data_path):
                for name in tqdm.tqdm(files):
                    if name.endswith('.jpg'):
                        self.data.append(os.path.join(root, name))

        print('finish generating background index, found texture image: {}'.format(len(self.data)))


#----------- Data Loader ----------#
#----------------------------------#
def data_loader(opts, shuffle=True):
    background_dataset = BackgroundDataset(opts, [opts.PRW_img_path])
    return DataLoader(dataset=background_dataset, batch_size=opts.batch_size, shuffle=shuffle,
                      num_workers=opts.n_data_workers, drop_last=True)
