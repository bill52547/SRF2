#!/usr/bin/env python
# encoding: utf-8
'''
@author: Minghao Guo, Xiang Hong, Gaoyu Chen and Weijie Tao
@license: LGPL_v3.0
@contact: mh.guo0111@gmail.com
@software: srf_v2
@file: psf.py
@date: 11/13/2018
@desc: new version of Scalable Reconstruction Framework for Medical Imaging
'''

import h5py
import numpy as np
import scipy.interpolate as interp
from scipy import sparse

from srf2.meta.image_meta import *
from srf2.meta.psf_meta import PsfMeta3d

__all__ = ('PSF_3d',)
_sqrt_pi = np.sqrt(np.pi)


class PSF_3d:
    def __init__(self, meta: PsfMeta3d = PsfMeta3d(), image_meta: Image_meta_3d =
    Image_meta_3d(), matrix_xy = None, matrix_z = None, matrix = None):
        self._meta = meta
        self._image_meta = image_meta
        if matrix_xy is None:
            self._matrix_xy = sparse.csr_matrix((image_meta.n_xy, image_meta.n_xy),
                                                dtype = np.float32)
            self._matrix_z = sparse.csr_matrix((image_meta.n_z, image_meta.n_z), dtype = np.float32)
            self.generate_separate_matrix()
        else:
            self._matrix_xy = matrix_xy
            self._matrix_z = matrix_z
        if matrix is not None:
            self._matrix = matrix
        else:
            self._matrix = sparse.csr_matrix((image_meta.n_all, image_meta.n_all),
                                             dtype = np.float32)

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

    @property
    def matrix(self):
        return self._matrix

    def save_h5(self, path = None, mode = 'w'):
        if path is None:
            path = 'tmp' + self.__class__.__name__ + '.h5'
        self.meta.save_h5(path, mode)
        self.image_meta.save_h5(path, mode)
        with h5py.File(path, 'r+') as fout:
            group = fout.create_group('PSF')
            row, col = self.matrix_xy.nonzero()
            data = self.matrix_xy.data
            group.create_dataset('_matrix_xy_row', data = row, compression = "gzip")
            group.create_dataset('_matrix_xy_col', data = col, compression = "gzip")
            group.create_dataset('_matrix_xy_data', data = data, compression = "gzip")
            row, col = self.matrix_z.nonzero()
            data = self.matrix_z.data
            group.create_dataset('_matrix_z_row', data = row, compression = "gzip")
            group.create_dataset('_matrix_z_col', data = col, compression = "gzip")
            group.create_dataset('_matrix_z_data', data = data, compression = "gzip")
            #
            # row, col = self.matrix.nonzero()
            # data = self.matrix.data
            # group.create_dataset('_matrix_full_row', data = row, compression = "gzip")
            # group.create_dataset('_matrix_full_col', data = col, compression = "gzip")
            # group.create_dataset('_matrix_full_data', data = data, compression = "gzip")

    @classmethod
    def load_h5(cls, path = None):
        if path is None:
            path = 'tmp' + cls.__name__ + '.h5'

        meta = PsfMeta3d.load_h5(path)
        image_meta = Image_meta_3d.load_h5(path)
        with h5py.File(path, 'r') as fin:
            dataset = fin['PSF']
            row = np.array(dataset['_matrix_xy_row'])
            col = np.array(dataset['_matrix_xy_col'])
            data = np.array(dataset['_matrix_xy_data'])
            matrix_xy = sparse.csr_matrix(((row, col), data),
                                          shape = (image_meta.n_xy, image_meta.n_xy),
                                          dtype = np.float32)

            row = np.array(dataset['_matrix_z_row'])
            col = np.array(dataset['_matrix_z_col'])
            data = np.array(dataset['_matrix_z_data'])
            matrix_z = sparse.csr_matrix(((row, col), data),
                                         shape = (image_meta.n_z, image_meta.n_z),
                                         dtype = np.float32)

            # row = np.array(dataset['_matrix_full_row'])
            # col = np.array(dataset['_matrix_full_col'])
            # data = np.array(dataset['_matrix_full_data'])
            # matrix_full = sparse.csr_matrix(((row, col), data),
            #                                 shape = (image_meta.n_z, image_meta.n_z),
            #                                 dtype = np.float32)
            return PSF_3d(meta, image_meta, matrix_xy, matrix_z)

    def _generate_matrix_xy_full(self):
        lil_xy = sparse.lil_matrix((self.image_meta.n_all, self.image_meta.n_all),
                                   dtype = np.float32)
        row, col = self.matrix_xy.nonzero()
        data = self.matrix_xy.data
        for iz in np.arange(self.image_meta.n_z):
            lil_xy[row * self.image_meta.n_z + iz, col * self.image_meta.n_z + iz] = data
        return lil_xy.tocsr()

    def _generate_matrix_z_full(self):
        lil_z = sparse.lil_matrix((self.image_meta.n_all, self.image_meta.n_all),
                                  dtype = np.float32)
        row, col = self.matrix_z.nonzero()
        data = self.matrix_z.data
        for ix in np.arange(self.image_meta.n_x):
            for iy in np.arange(self.image_meta.n_y):
                ind = iy + ix * self.image_meta.n_y
                lil_z[row + self.image_meta.n_z * ind, col + self.image_meta.n_z * ind] = data
        return lil_z.tocsr()

    def generate_matrix_full(self):
        print('Generating full PSF matrix')
        return self._generate_matrix_xy_full() * self._generate_matrix_z_full()

    def generate_separate_matrix(self):
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


class PSF_2d:
    pass


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
