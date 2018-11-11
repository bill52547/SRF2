# import attr
# import numpy as np
# import h5py
# from srf2.data.image import *
# import scipy.optimize as opt
# import scipy.interpolate as interp
# from scipy import sparse
# from tqdm import tqdm
#
# __all__ = ['PSF_meta', 'PSF']
#
#
# @attr.s
# class PSF_meta:
#     image_filename_prefix = attr.ib(default='.')
#     num = attr.ib(default=0)
#     pos = attr.ib(default=np.zeros(0, ))
#
#     is_sigma0 = attr.ib(default=False)
#     is_sigma = attr.ib(default=False)
#     is_matrix = attr.ib(default=False)
#     _threshold = attr.ib(default=1e-4)
#
#     @property
#     def threshold(self):
#         return self._threshold
#
#     def save_h5(self, path, mode='w'):
#         dt = h5py.special_dtype(vlen=str)
#         with h5py.File(path, mode) as fout:
#             group = fout.create_group('psf_meta')
#             group.attrs.create('_threshold', data=self._threshold)
#             group.attrs.create('is_sigma0', data=1 if self.is_sigma0 else 0)
#             group.attrs.create('is_sigma', data=1 if self.is_sigma else 0)
#             group.attrs.create('is_matrix', data=1 if self.is_matrix else 0)
#             dset = group.create_dataset('image_filename_prefix', (1,), dtype=dt)
#             dset[0] = self.image_filename_prefix
#             group.attrs.create('num', data=self.num)
#             group.attrs.create('pos', data=self.pos)
#
#     def load_h5(path):
#         with h5py.File(path, 'r') as fin:
#             group = fin['psf_meta']
#             _threshold = np.float32(group.attrs['_threshold'])
#             is_sigma0 = bool(group.attrs['is_sigma0'])
#             is_sigma = bool(group.attrs['is_sigma'])
#             is_matrix = bool(group.attrs['is_matrix'])
#             image_filename_prefix = group['image_filename_prefix'][0]
#             num = np.int32(group.attrs['num'])
#             pos = np.array(group.attrs['pos'])
#             return PSF_meta(_threshold, is_sigma0, is_sigma, is_matrix, image_filename_prefix, num, pos)
#
#
# @attr.s
# class PSF:
#     image_meta = attr.ib()
#     meta = attr.ib()
#
#     _sigma_x0 = attr.ib(default=None)
#     _sigma_y0 = attr.ib(default=None)
#     _sigma_z0 = attr.ib(default=None)
#
#     _sigma_x = attr.ib(default=None)
#     _sigma_y = attr.ib(default=None)
#     _sigma_z = attr.ib(default=None)
#
#     _matrix_xy = attr.ib(default=sparse.csr_matrix((1, 1), dtype=np.float32))
#     _matrix_z = attr.ib(default=sparse.csr_matrix((1, 1), dtype=np.float32))
#
#     @property
#     def sigma_x0(self):
#         return self._sigma_x0
#
#     @property
#     def sigma_y0(self):
#         return self._sigma_y0
#
#     @property
#     def sigma_z0(self):
#         return self._sigma_z0
#
#     @property
#     def sigma_x(self):
#         return self._sigma_x
#
#     @property
#     def sigma_y(self):
#         return self._sigma_y
#
#     @property
#     def sigma_z(self):
#         return self._sigma_z
#
#     @property
#     def matrix_xy(self):
#         return self._matrix_xy
#
#     @property
#     def matrix_z(self):
#         return self._matrix_z
#
#     def generate_sigma0(self):
#         if self.meta.is_sigma0:
#             return self
#         print("fitting parameters of PSF")
#         for i in tqdm(range(self.meta.num)):
#             img = Image.load_h5(self.meta.image_filename_prefix + str(i))
#             img_meta = Image_meta.load_h5(self.meta.image_filename_prefix + str(i))
#             pos = self.meta.pos[i]
#             img = img / np.sum(img)
#             x1, y1, z1 = img_meta.meshgrid()
#             x1, y1, z1 = x1.flatten() - pos[0], y1.flatten() - pos[1], z1.flatten() - pos[2]
#             theta = 0 * img_meta.theta()
#             popt, pcov = opt.curve_fit(_gaussian_3d, (x1, y1, z1, theta), img.flatten())
#             self._sigma_x0[i] = popt[0]
#             self._sigma_y0[i] = popt[1]
#             self._sigma_z0[i] = popt[2]
#
#         self.meta.is_sigma0 = True
#
#     def generate_sigma(self):
#         if self.meta.is_sigma:
#             return self
#         if not self.meta.is_sigma0:
#             self.generate_sigma0()
#         x1, y1, z1 = self.image_meta.meshgrid()
#         R1 = np.sqrt(x1 ** 2 + y1 ** 2)
#         z1 = np.abs(z1)
#         R0 = np.sqrt(self.meta.pos[0] ** 2 + self.meta.pos[1] ** 2)
#         z0 = np.abs(self.meta.pos[2])
#         fsigx = interp.interp1d(R0, self._sigma_x0, kind='quadratic', fill_value='extrapolate')
#         fsigy = interp.interp1d(R0, self._sigma_y0, kind='quadratic', fill_value='extrapolate')
#         fsigz = interp.interp1d(z0, self._sigma_z0, kind='quadratic', fill_value='extrapolate')
#         self._sigma_x = fsigx(R1)
#         self._sigma_y = fsigy(R1)
#         self._sigma_z = fsigz(z1)
#
#         self.meta.is_sigma = True
#
#     def generate_matrix(self):
#         if self.meta.is_matrix:
#             return self
#
#         if not self.meta.is_sigma:
#             self.generate_sigma()
#
#         x1, y1 = self.image_meta.meshgrid_2d()
#         lil_matrix_xy = sparse.lil_matrix((self.image_meta.n_xy, self.image_meta.n_xy), dtype=np.float32)
#         lil_matrix_z = sparse.lil_matrix((self.image_meta.n_z, self.image_meta.n_z), dtype=np.float32)
#
#         theta = self.image_meta.theta()
#         print('Generating PSF matrix xy')
#         for ix in tqdm(np.arange(self.image_meta.shape[0])):
#             for iy in np.arange(self.image_meta.shape[1]):
#                 ind = ix + iy * self.image_meta.shape[0]
#
#                 img_tmp = _gaussian_2d((x1 - x1[ix, iy], y1 - y1[ix, iy], theta[ix, iy]), self._sigma_x, self._sigma_y)
#                 amp = 1 / 2 / np.pi / self._sigma_x / self._sigma_y
#                 col = y1[img_tmp > amp * self.meta.threshold] + x1[img_tmp > amp * self.meta.threshold] * \
#                       self.image_meta.shape[1]
#                 row = np.ones(col.shape) * ind
#                 data = img_tmp[img_tmp > amp * self.meta.threshold]
#                 lil_matrix_xy[row, col] = data
#         self._matrix_xy = lil_matrix_xy.tocsr()
#
#         z1 = np.arange(self.image_meta.shape[2]) * self.image_meta.unit_size[2] + self.image_meta.center[2] - \
#              self.image_meta.size[2] / 2 + self.image_meta.unit_size[2] / 2
#
#         print('Generating PSF matrix z')
#         for iz in tqdm(np.arange(self.image_meta.shape[2])):
#             img_tmp = _gaussian_1d(z1 - z1[iz], self._sigma_z)
#             amp = 1 / np.sqrt(2 * np.pi) / self._sigma_z
#             col = x1[img_tmp > amp * self.meta.threshold]
#             row = np.ones(col.shape) * iz
#             data = img_tmp[img_tmp > amp * self.meta.threshold]
#             lil_matrix_z[row, col] = data
#         self._matrix_xy = lil_matrix_xy.tocsr()
#
#         self.meta.is_matrix = True
#
#     def psf_transform(self, _image: Image):
#         if self.image_meta != _image.meta:
#             raise TypeError('Input image object has different implementing meta from the image_meta in PSF object')
#
#         result = Image(_image.meta, _image.data)
#
#         print('imaging is transforming by PSF_z')
#
#         for ix in tqdm(range(self.image_meta.shape[0])):
#             for iy in range(self.image_meta.shape[1]):
#                 result[ix, iy, :] = self._matrix_z * result[ix, iy, :]
#
#         result2 = result.transpose()
#
#         print('imaging is transforming by PSF_xy')
#         for iz in tqdm(range(self.image_meta.n_z)):
#             result2[:, :, iz] = (self._matrix_xy * result2[:, :, iz].flatten()).reshape(result2.meta.shape[:2])
#
#         return result2.transpose()
#
#     def save_h5(self, path):
#         self.image_meta.save_h5(path)
#         self.meta.save_h5(path, 'r+')
#         with h5py.File(path, 'r+') as fout:
#             group = fout.create_group('psf')
#             if self.meta.is_sigma0:
#                 group.create_dataset('_sigma_x0', data=self._sigma_x0)
#                 group.create_dataset('_sigma_y0', data=self._sigma_y0)
#                 group.create_dataset('_sigma_z0', data=self._sigma_z0)
#             if self.meta.is_sigma:
#                 group.create_dataset('_sigma_x', data=self._sigma_x)
#                 group.create_dataset('_sigma_y', data=self._sigma_y)
#                 group.create_dataset('_sigma_z', data=self._sigma_z)
#             if self.meta.is_matrix:
#                 row, col = self._matrix_xy.nonzero()
#                 group.create_dataset('_matrix_xy_row', data=row)
#                 group.create_dataset('_matrix_xy_col', data=col)
#                 group.create_dataset('_matrix_xy_data', data=self._matrix_xy.data)
#
#                 row, col = self._matrix_z.nonzero()
#                 group.create_dataset('_matrix_z_row', data=row)
#                 group.create_dataset('_matrix_z_col', data=col)
#                 group.create_dataset('_matrix_z_data', data=self._matrix_z.data)
#
#     def load_h5(path):
#         img_meta = Image_meta.load_h5(path)
#         meta = PSF_meta.load_h5(path)
#         with h5py.File(path, 'r') as fin:
#             group = fin['psf']
#             if meta.is_sigma0:
#                 _sigma_x0 = group['_sigma_x0']
#                 _sigma_y0 = group['_sigma_y0']
#                 _sigma_z0 = group['_sigma_z0']
#             else:
#                 _sigma_x0 = _sigma_y0 = _sigma_z0 = np.ones(meta.num, ) * 1e8
#
#             if meta.is_sigma:
#                 _sigma_x = group['_sigma_x']
#                 _sigma_y = group['_sigma_y']
#                 _sigma_z = group['_sigma_z']
#             else:
#                 _sigma_x = _sigma_y = _sigma_z = np.ones(meta.num, ) * 1e8
#
#             if meta.is_matrix:
#                 row = group['_matrix_xy_row']
#                 col = group['_matrix_xy_col']
#                 data = group['_matrix_xy_data']
#                 _matrix_xy = sparse.csr_matrix((data, (row, col)), shape=(img_meta.n_xy, img_meta.n_xy),
#                                                dtype=np.float32)
#                 row = group['_matrix_z_row']
#                 col = group['_matrix_z_col']
#                 data = group['_matrix_z_data']
#                 _matrix_z = sparse.csr_matrix((data, (row, col)), shape=(img_meta.n_z, img_meta.n_z),
#                                               dtype=np.float32)
#             else:
#                 _matrix_xy = sparse.csr_matrix(shape=(img_meta.n_xy, img_meta.n_xy), dtype=np.float32)
#                 _matrix_z = sparse.csr_matrix(shape=(img_meta.n_z, img_meta.n_z), dtype=np.float32)
#
#             return PSF(img_meta, meta, _sigma_x0, _sigma_y0, _sigma_z0, _sigma_x, _sigma_y, _sigma_z, _matrix_xy,
#                        _matrix_xy)
#
#
# def _gaussian_1d(z, sigz):
#     return 1 / np.sqrt(2 * np.pi) / sigz * np.exp(-z ** 2 / 2 / sigz ** 2)
#
#
# def _gaussian_2d(x_y_t, sigx, sigy):
#     x = x_y_t[0]
#     y = x_y_t[1]
#     theta = x_y_t[2]
#
#     x1 = x * np.cos(theta) + y * np.sin(theta)
#     y1 = -x * np.sin(theta) + y * np.cos(theta)
#     return 1 / np.sqrt(2 * np.pi) ** 2 / sigx / sigy * np.exp(
#         -x1 ** 2 / 2 / sigx ** 2 - y1 ** 2 / 2 / sigy ** 2)
#
#
# def _gaussian_3d(x_y_z_t, sigx, sigy, sigz):
#     x = x_y_z_t[0]
#     y = x_y_z_t[1]
#     z = x_y_z_t[2]
#     theta = x_y_z_t[3]
#
#     x1 = x * np.cos(theta) + y * np.sin(theta)
#     y1 = -x * np.sin(theta) + y * np.cos(theta)
#     return 1 / np.sqrt(2 * np.pi) ** 3 / sigx / sigy / sigz * np.exp(-z ** 2 / 2 / sigz ** 2) * np.exp(
#         -x1 ** 2 / 2 / sigx ** 2 - y1 ** 2 / 2 / sigy ** 2)
