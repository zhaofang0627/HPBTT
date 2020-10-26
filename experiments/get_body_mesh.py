__author__ = 'matt'

from copy import deepcopy
from os.path import join, split, exists
import numpy as np
import cv2

from chumpy.utils import row, col


def get_body_mesh(obj_path, trans, rotation):
    from opendr.serialization import load_mesh

    from copy import deepcopy

    fname = obj_path
    mesh = load_mesh(fname)

    mesh.v = np.asarray(mesh.v, order='C')
    mesh.vc = mesh.v*0 + 1
    mesh.v -= row(np.mean(mesh.v, axis=0))
    mesh.v /= np.max(mesh.v)
    mesh.v *= 2.0
    get_body_mesh.mesh = mesh

    mesh = deepcopy(get_body_mesh.mesh)
    mesh.v = mesh.v.dot(cv2.Rodrigues(np.asarray(np.array(rotation), np.float64))[0])
    mesh.v = mesh.v + row(np.asarray(trans))
    return mesh


#
# def process(im, vmin, vmax):
#     shape = im.shape
#     im = deepcopy(im).flatten()
#     im[im>vmax] = vmax
#     im[im<vmin] = vmin
#     im -= vmin
#     im /= (vmax-vmin)
#     im = im.reshape(shape)
#     return im
