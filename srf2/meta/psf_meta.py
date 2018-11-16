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
from srf2.data.image import Image_3d

__all__ = ('PsfMeta3d', 'PsfMeta2d',)

_sqrt_2_pi = np.sqrt(np.pi * 2)


class PsfMeta3d(Meta):
    def __init__(self, mu = tuple([]), sigma = tuple([])):
        # if mu.shape != sigma.shape:
        #     raise ValueError('mu and sigma should have same shape')
        # if mu.size > 0 and mu.shape[1] != 3:
        #     raise ValueError('Only support 3D case in this class. Please go to PsfMeta2d')
        self._mu = np.array(mu, dtype = np.float32)
        self._sigma = np.array(sigma, dtype = np.float32)

    @property
    def sigma(self):
        return self._sigma

    @property
    def mu(self):
        return self._mu

    def add_para_xy(self, image, pos = (0, 0, 0), rang = 20):
        if isinstance(image, Image_3d):
            image = image.transpose(('x', 'y', 'z'))
            ix1, iy1, iz1 = image.meta.locate(pos)
            slice_x = slice_y = slice(None, None)
            slice_z = slice(int(np.round(iz1) - rang / image.meta.unit_size[2]),
                            int(np.round(iz1) + rang / image.meta.unit_size[2]) + 1)

            x1, y1 = image.meta.grid_centers_2d([slice_x, slice_y, slice_z])
            x1, y1 = x1 - pos[0], y1 - pos[1]
            # print(np.max(x1))
            # print(np.max(y1))
            image_new_data = np.sum(image[slice_x, slice_y, slice_z].normalize().data, axis = 2) \
                             / image.meta.unit_size[0] / image.meta.unit_size[1]

            p = _fitgaussian_2d(image_new_data, x1, y1)

            out_args = np.append(np.abs(np.array(p[:2])), [0])
            if self._mu.size == 0:
                self._mu = np.array(pos)
                self._sigma = np.array([out_args])
            else:
                self._mu = np.vstack((self._mu, [pos]))
                self._sigma = np.vstack((self._sigma, out_args))
            return x1, y1, image_new_data, p[:2]
        else:
            raise NotImplementedError

    def add_para_z(self, image, pos = (0, 0, 0), rang = 20):
        if isinstance(image, Image_3d):
            func = _gaussian_1d
            image = image.transpose(('x', 'y', 'z'))
            ix1, iy1, iz1 = image.meta.locate(pos)
            slice_x = slice(int(np.round(ix1) - rang / image.meta.unit_size[0]),
                            int(np.round(ix1) + rang / image.meta.unit_size[0]) + 1)
            slice_y = slice(int(np.round(iy1) - rang / image.meta.unit_size[1]),
                            int(np.round(iy1) + rang / image.meta.unit_size[1]) + 1)
            slice_z = slice(None, None)

            z1 = image.meta.grid_centers_1d([slice_x, slice_y, slice_z]) - pos[2]
            image_new_data = np.sum(image[slice_x, slice_y, slice_z].normalize().data,
                                    axis = (0, 1)) / image.meta.unit_size[2]
            p = _fitgaussian_1d(image_new_data, z1)

            out_args = np.append([0, 0], np.abs(np.array(p[0])))
            if self._mu.size == 0:
                self._mu = np.array(pos)
                self._sigma = np.array([out_args])
            else:
                # self._mu = np.vstack((self._mu, pos[:2] + tuple(p[1])))
                self._mu = np.vstack((self._mu, pos))
                self._sigma = np.vstack((self._sigma, out_args))
            return z1, image_new_data, p[:1]

        else:
            raise NotImplementedError
    # def add_para(self, image, pos = (0, 0, 0), rang = 20):
    #     if isinstance(image, Image_3d):
    #         image = image.transpose(('x', 'y', 'z'))
    #         ix1, iy1, iz1 = image.meta.locate(pos)
    #         slice_x = slice(int(ix1 - rang / image.meta.unit_size[0]),
    #                         int(ix1 + rang / image.meta.unit_size[0]) + 1)
    #         slice_y = slice(int(iy1 - rang / image.meta.unit_size[1]),
    #                         int(iy1 + rang / image.meta.unit_size[1]) + 1)
    #         slice_z = slice(int(iz1 - rang / image.meta.unit_size[2]),
    #                         int(iz1 + rang / image.meta.unit_size[2]) + 1)
    #
    #         x1, y1, z1 = image.meta.meshgrid([slice_x, slice_y, slice_z])
    #         x1 /= image.meta.unit_size[0]
    #         y1 /= image.meta.unit_size[1]
    #         z1 /= image.meta.unit_size[2]
    #
    #         image_new_data = image[slice_x, slice_y, slice_z].normalize().data
    #
    #         p = _fit_gaussian_3d(image_new_data, x1, y1, z1)
    #
    #         out_args = np.abs(np.array(p[:3]))
    #         if self._mu.size == 0:
    #             self._mu = np.array(pos)
    #             self._sigma = np.array([out_args])
    #         else:
    #             self._mu = np.vstack((self._mu, np.array(pos)))
    #             self._sigma = np.vstack((self._sigma, out_args))
    #         return x1, y1, image_new_data, p[:3]
    #     else:
    #         raise NotImplementedError


class PsfMeta2d(Meta):
    pass


def _gaussian_1d(sigz):
    sigz = abs(sigz)

    return lambda z: 1 / _sqrt_2_pi / sigz * np.exp(-z ** 2 / 2 / sigz ** 2)


def _gaussian_2d(sigx, sigy):
    sigx = abs(sigx)
    sigy = abs(sigy)
    return lambda x, y: 1 / _sqrt_2_pi ** 2 / sigx / sigy * np.exp(
        -x ** 2 / 2 / sigx ** 2) * np.exp(-y ** 2 / 2 / sigy ** 2)


def _gaussian_3d_xy(x_y_z_t, sigx, sigy):
    x = x_y_z_t[0]
    y = x_y_z_t[1]
    z = x_y_z_t[2]
    t = x_y_z_t[3]
    sigx = abs(sigx)
    sigy = abs(sigy)

    x1 = x * np.cos(t) + y * np.sin(t)
    y1 = -x * np.sin(t) + y * np.cos(t)

    return 1 / _sqrt_2_pi ** 2 / sigx / sigy * np.exp(-x1 ** 2 / 2 / sigx ** 2) * np.exp(
        - y1 ** 2 / 2 / sigy ** 2)


def _gaussian_3d(sigx, sigy, sigz):
    sigx = abs(sigx)
    sigy = abs(sigy)
    sigz = abs(sigz)

    return lambda x, y, z: 1 / _sqrt_2_pi ** 3 / sigx / sigy / sigz * np.exp(
        -x ** 2 / 2 / sigx ** 2) * np.exp(- y ** 2 / 2 / sigy ** 2) * np.exp(
        -z ** 2 / 2 / sigz ** 2)


def _fitgaussian_2d(data, x, y):
    def _error_function(p):
        return np.ravel(_gaussian_2d(*p)(x, y) - data)

    return opt.leastsq(_error_function, np.array([1, 1]))


def _fitgaussian_1d(data, x):
    def _error_function(p):
        return np.ravel(_gaussian_2d(*p)(x) - data)

    return opt.leastsq(_error_function, np.array([1]))


def _fit_gaussian_3d(data, x, y, z):
    def _error_function(p):
        return np.ravel(_gaussian_2d(*p)(x, y, z) - data)

    return opt.leastsq(_error_function, np.array([1, 1, 1]))

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
