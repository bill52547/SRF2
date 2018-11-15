#!/usr/bin/env python
# encoding: utf-8
'''
@author: Minghao Guo, Xiang Hong, Gaoyu Chen and Weijie Tao
@license: LGPL_v3.0
@contact: mh.guo0111@gmail.com
@software: srf_v2
@file: psf_meta.py
@date: 11/11/2018
@desc: new version of Scalable Reconstruction Framework for Medical Imaging
'''

import numpy as np
import scipy.optimize as opt

from srf2.core.abstracts import Meta
from srf2.data.image import *

__all__ = ('PsfMeta3d', 'PsfMeta2d',)

_sqrt_pi = np.sqrt(np.pi)


class PsfMeta3d(Meta):
    def __init__(self, mu: np.ndarray = np.array([]), sigma: np.ndarray = np.array([])):
        if mu.shape != sigma.shape:
            raise ValueError('mu and sigma should have same shape')
        if mu.size > 0 and mu.shape[1] != 3:
            raise ValueError('Only support 3D case in this class. Please go to PsfMeta2d')
        self._mu = mu
        self._sigma = sigma

    @property
    def sigma(self):
        return self._sigma

    @property
    def mu(self):
        return self._mu

    def add_para_xy(self, image, pos = (0, 0), rang = 20):
        if isinstance(image, Image_3d):
            func = _gaussian_2d
            image = image.transpose(('x', 'y', 'z'))
            ix1, iy1, iz1 = image.meta.locate(pos)
            slice_x = slice_y = slice(None, None)
            slice_z = slice(int(iz1 - rang / image.meta.unit_size[2]),
                            int(iz1 + rang / image.meta.unit_size[2]) + 1)

            x1, y1 = image.meta.meshgrid_2d([slice_x, slice_y, slice_z])
            x1, y1 = x1 - pos[0], y1 - pos[1]
            image_new_data = np.sum(image[slice_x, slice_y, slice_z].normalize().data, axis = 2)
            popt, pcov = opt.curve_fit(func, (x1.flatten(), y1.flatten(), 0),
                                       image_new_data.flatten())

            out_args = np.abs(np.array(popt[:2] + [0]))
            if self._mu.size == 0:
                self._mu = np.array(pos)
                self._sigma = out_args
            else:
                self._mu = np.vstack((self._mu, np.array(pos)))
                self._sigma = np.vstack((self._sigma, out_args))
            return x1, y1, image_new_data, popt[:2]
        else:
            raise NotImplementedError

    def add_para_z(self, image, pos = (0, 0), rang = 20):
        if isinstance(image, Image_3d):
            func = _gaussian_1d
            image = image.transpose(('x', 'y', 'z'))
            ix1, iy1, iz1 = image.meta.locate(pos)
            slice_x = slice(int(ix1 - rang / image.meta.unit_size[0]),
                            int(ix1 + rang / image.meta.unit_size[0]) + 1)
            slice_y = slice(int(iy1 - rang / image.meta.unit_size[1]),
                            int(iy1 + rang / image.meta.unit_size[1]) + 1)
            slice_z = slice(None, None)

            z1 = image.meta.meshgrid_1d() - pos[2]
            image_new_data = np.sum(image[slice_x, slice_y, slice_z].normalize().data, axis = (0,
                                                                                               1))
            popt, pcov = opt.curve_fit(func, z1, image_new_data)
            out_args = np.append([0, 0], np.abs(np.array(popt[:1])))
            if self._mu.size == 0:
                self._mu = np.array(pos)
                self._sigma = out_args
            else:
                self._mu = np.vstack((self._mu, np.array(pos)))
                self._sigma = np.vstack((self._sigma, out_args))
            return z1, image_new_data, popt[:2]

        else:
            raise NotImplementedError


class PsfMeta2d(Meta):
    pass


def _gaussian_1d(z, sigz):
    sigz = abs(sigz)

    return 1 / _sqrt_pi / sigz * np.exp(-z ** 2 / 2 / sigz ** 2)


def _gaussian_2d(x_y_t, sigx, sigy):
    x = x_y_t[0]
    y = x_y_t[1]
    theta = x_y_t[2]
    sigx = abs(sigx)
    sigy = abs(sigy)
    x1 = x * np.cos(theta) + y * np.sin(theta)
    y1 = -x * np.sin(theta) + y * np.cos(theta)
    return 1 / _sqrt_pi ** 2 / sigx / sigy * np.exp(-x1 ** 2 / 2 / sigx ** 2) * np.exp(- y1 ** 2
                                                                                       / 2 / sigy ** 2)


def _gaussian_3d_xy(x_y_z_t, sigx, sigy):
    x = x_y_z_t[0]
    y = x_y_z_t[1]
    z = x_y_z_t[2]
    t = x_y_z_t[3]
    sigx = abs(sigx)
    sigy = abs(sigy)

    x1 = x * np.cos(t) + y * np.sin(t)
    y1 = -x * np.sin(t) + y * np.cos(t)

    return 1 / _sqrt_pi ** 2 / sigx / sigy * np.exp(-x1 ** 2 / 2 / sigx ** 2) * np.exp(
        - y1 ** 2 / 2 / sigy ** 2)


def _gaussian_3d(x_y_z_t, sigx, sigy, sigz):
    x = x_y_z_t[0]
    y = x_y_z_t[1]
    z = x_y_z_t[2]
    t = x_y_z_t[3]

    sigx = abs(sigx)
    sigy = abs(sigy)
    sigz = abs(sigz)

    x1 = x * np.cos(t) + y * np.sin(t)
    y1 = -x * np.sin(t) + y * np.cos(t)
    return 1 / _sqrt_pi ** 3 / sigx / sigy / sigz * np.exp(-x1 ** 2 / 2 / sigx ** 2) * np.exp(
        - y1 ** 2 / 2 / sigy ** 2) * np.exp(-z ** 2 / 2 / sigz ** 2)

#
# def _gaussian_3d(x_y_z_t, mux, muy, muz, sigx, sigy, sigz):
#     x = x_y_z_t[0]
#     y = x_y_z_t[1]
#     z = x_y_z_t[2]
#     t = x_y_z_t[3]
#     sigx = abs(sigx)
#     sigy = abs(sigy)
#     sigz = abs(sigz)
#
#     x1 = x * np.cos(t) + y * np.sin(t) - mux
#     y1 = -x * np.sin(t) + y * np.cos(t) - muy
#     z -= muz
#     return 1 / _sqrt_pi ** 3 / sigx / sigy / sigz * np.exp(-x1 ** 2 / 2 / sigx ** 2) * np.exp(
#         - y1 ** 2 / 2 / sigy ** 2) * np.exp(-z ** 2 / 2 / sigz ** 2)
