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
import scipy.interpolate as interp
import scipy.optimize as opt
from scipy import sparse

from srf2.core.abstracts import Meta
from srf2.data.image import *
from srf2.meta.image_meta import *

__all__ = ('PSF_meta_3d', 'PSF_3d',)
_sqrt_pi = np.sqrt(np.pi)


class PSF_meta_3d(Meta):
    def __init__(self, mu: np.ndarray = np.array([]), sigma: np.ndarray = np.array([])):
        if mu.shape != sigma.shape:
            raise ValueError('mu and sigma should have same shape')
        if mu.size > 0 and mu.shape[1] != 3:
            raise ValueError('Only support 3D case in this class. Please go to PSF_meta_2d')
        self._mu = mu
        self._sigma = sigma

    @property
    def sigma(self):
        return self._sigma

    @property
    def mu(self):
        return self._mu

    def add_para(self, image: Image_3d, pos = (0, 0, 0), rang = 20):
        if isinstance(image, Image_3d):
            func = _gaussian_3d
            image = image.transpose(('x', 'y', 'z'))
            ix1, iy1, iz1 = image.meta.locate(pos)
            slice_x = slice(int(ix1 - rang / image.meta.unit_size[0]),
                            int(ix1 + rang / image.meta.unit_size[0]))
            slice_y = slice(int(iy1 - rang / image.meta.unit_size[1]),
                            int(iy1 + rang / image.meta.unit_size[1]))
            slice_z = slice(int(iz1 - rang / image.meta.unit_size[2]),
                            int(iz1 + rang / image.meta.unit_size[2]))

            x1, y1, z1 = image.meta.meshgrid([slice_x, slice_y, slice_z])
            x1, y1, z1 = x1.flatten() - pos[0], y1.flatten() - pos[1], z1.flatten() - pos[2]
            popt, pcov = opt.curve_fit(func, (x1, y1, z1, 0),
                                       image[slice_x, slice_y, slice_z].normalize().data.flatten())
            if self._mu.size == 0:
                self._mu = np.array(pos)
                self._sigma = np.array(popt[:3])
            else:
                self._mu = np.vstack((self._mu, np.array(pos)))
                self._sigma = np.vstack((self._sigma, np.array(popt[:3])))
        else:
            raise NotImplementedError


class PSF_meta_2d(Meta):
    pass


class PSF_matrix:
    pass


class PSF_3d:
    def __init__(self, meta: PSF_meta_3d = PSF_meta_3d(), image_meta: Image_meta_3d =
    Image_meta_3d()):
        self._meta = meta
        self._image_meta = image_meta
        self._matrix_xy = sparse.csr_matrix((image_meta.n_xy, image_meta.n_xy), dtype = np.float32)
        self._matrix_z = sparse.csr_matrix((image_meta.n_z, image_meta.n_z), dtype = np.float32)
        self.generate_matrix()

    @property
    def meta(self):
        return self._meta

    @property
    def image_meta(self):
        return self._image_meta

    @property
    def matrix_xy(self):
        return self._matrix_xy

    @property
    def matrix_z(self):
        return self._matrix_z

    def matrix_xy_full(self):
        lil_xy = sparse.lil_matrix((self.image_meta.n_all, self.image_meta.n_all),
                                   dtype = np.float32)
        row, col = self.matrix_xy.nonzero()
        data = self.matrix_xy.data
        for iz in np.arange(self.image_meta.n_z):
            lil_xy[row * self.image_meta.n_z + iz, col * self.image_meta.n_z + iz] = data
        return lil_xy.tocsr()

    def matrix_z_full(self):
        lil_z = sparse.lil_matrix((self.image_meta.n_all, self.image_meta.n_all),
                                  dtype = np.float32)
        row, col = self.matrix_z.nonzero()
        data = self.matrix_z.data
        for ix in np.arange(self.image_meta.n_x):
            for iy in np.arange(self.image_meta.n_y):
                ind = iy + ix * self.image_meta.n_y
                lil_z[row + self.image_meta.n_z * ind, col + self.image_meta.n_z * ind] = data
        return lil_z.tocsr()

    def matrix(self):
        print('Generating full PSF matrix')
        return self.matrix_xy_full() * self.matrix_z_full()

    def generate_matrix(self):
        x1, y1 = self.image_meta.meshgrid_2d()
        z1 = np.arange(self.image_meta.n_z)
        R1 = np.sqrt(x1 ** 2 + y1 ** 2)
        z1 = np.abs(z1)
        R0 = np.sqrt(self.meta.mu[:, 0] ** 2 + self.meta.mu[:, 1] ** 2)
        z0 = np.abs(self.meta.mu[:, 2])

        ind_xy = np.where(R0 > 0)[0]
        ind_z = np.where(z0 > 0)[0]

        if ind_xy.size > 1:
            fsigx = interp.interp1d(R0[ind_xy], self.meta.sigma[ind_xy, 0], kind = 'quadratic',
                                    fill_value = 'extrapolate')
            fsigy = interp.interp1d(R0[ind_xy], self.meta.sigma[ind_xy, 1], kind = 'quadratic',
                                    fill_value = 'extrapolate')
            sigma_x, sigma_y = fsigx(R1), fsigy(R1)
        else:
            sigma_x = np.mean(self.meta.sigma[:, 0]) * np.ones(R1.shape)
            sigma_y = np.mean(self.meta.sigma[:, 1]) * np.ones(R1.shape)

        if ind_z.size > 1:
            fsigz = interp.interp1d(z0[ind_z], self.meta.sigma[ind_z, 2], kind = 'quadratic',
                                    fill_value = 'extrapolate')
            sigma_z = fsigz(z1)
        else:
            sigma_z = np.mean(self.meta.sigma[:, 2]) * np.ones(z1.shape)

        lil_matrix_xy = sparse.lil_matrix((self.image_meta.n_xy, self.image_meta.n_xy),
                                          dtype = np.float32)
        lil_matrix_z = sparse.lil_matrix((self.image_meta.n_z, self.image_meta.n_z),
                                         dtype = np.float32)
        theta = self.image_meta.theta()
        for ix in np.arange(self.image_meta.n_x):
            for iy in np.arange(self.image_meta.n_y):
                ind = iy + ix * self.image_meta.n_y
                img_tmp = _gaussian_2d((x1 - x1[ix, iy], y1 - y1[ix, iy], theta[ix, iy]),
                                       sigma_x[ix, iy], sigma_y[ix, iy])
                gk = img_tmp.flatten()
                row = np.where(gk > 0)[0]
                col = ind * np.ones(row.shape)
                data = gk[row]
                lil_matrix_xy[row, col] = data
        self._matrix_xy = lil_matrix_xy.tocsr()

        for iz in np.arange(self.image_meta.n_z):
            img_tmp = _gaussian_1d(z1 - z1[iz], sigma_z[iz])
            gk = img_tmp.flatten()
            row = np.where(gk > 0)[0]
            col = iz * np.ones(row.shape)
            data = gk[row]
            lil_matrix_z[row, col] = data

        self._matrix_z = lil_matrix_z.tocsr()


def _gaussian_1d(z, sigz):
    return 1 / _sqrt_pi / sigz * np.exp(-z ** 2 / 2 / sigz ** 2)


def _gaussian_2d(x_y_t, sigx, sigy):
    x = x_y_t[0]
    y = x_y_t[1]
    theta = x_y_t[2]

    x1 = x * np.cos(theta) + y * np.sin(theta)
    y1 = -x * np.sin(theta) + y * np.cos(theta)
    return 1 / _sqrt_pi ** 2 / sigx / sigy * np.exp(
        -x1 ** 2 / 2 / sigx ** 2 - y1 ** 2 / 2 / sigy ** 2)


def _gaussian_3d(x_y_z_t, sigx, sigy, sigz):
    x = x_y_z_t[0]
    y = x_y_z_t[1]
    z = x_y_z_t[2]
    t = x_y_z_t[3]

    x1 = x * np.cos(t) + y * np.sin(t)
    y1 = -x * np.sin(t) + y * np.cos(t)

    return 1 / _sqrt_pi ** 3 / sigx / sigy / sigz * np.exp(-x1 ** 2 / 2 / sigx ** 2) * np.exp(
        - y1 ** 2 / 2 / sigy ** 2) * np.exp(-z ** 2 / 2 / sigz ** 2)
