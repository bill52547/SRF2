# import numpy as np
# from scipy import sparse
# import h5py
# import attr
#
# __all__ = ['Projection_meta', 'Sinogram_projection_meta', 'Listmode_projection_meta', 'Projection',
#            'Sinogram_projection', 'Listmode_projection']
#
#
# @attr.s
# class Projection_meta:
#     _shape = attr.ib(default = (1,) * 3)
#     _center = attr.ib(default = (0,) * 3)
#     _size = attr.ib(default = (1,) * 3)
#     _inner_radius = attr.ib(default = 99.0)
#     _outer_radius = attr.ib(default = 119.0)
#     _axial_length = attr.ib(default = 33.4)
#     _nb_rings = attr.ib(default = 1, converter = int)
#     _nb_blocks_per_ring = attr.ib(default = 16, converter = int)
#     _gap = attr.ib(default = 0.0)
#
#     @property
#     def shape(self):
#         return self._shape
#
#     @property
#     def center(self):
#         return self._center
#
#     @property
#     def size(self):
#         return self._size
#
#     @property
#     def unit_size(self):
#         return tuple(ti / tj for (ti, tj) in zip(self._size, self._shape))
#
#     @property
#     def inner_radius(self) -> float:
#         return self._inner_radius
#
#     @property
#     def outer_radius(self) -> float:
#         return self._outer_radius
#
#     @property
#     def axial_length(self) -> float:
#         return self._axial_length
#
#     @property
#     def nb_rings(self):
#         return self._nb_rings
#
#     @property
#     def nb_blocks_per_ring(self):
#         return self._nb_blocks_per_ring
#
#     @property
#     def gap(self):
#         return self._gap
#
#     @property
#     def n_sensors(self):
#         return self._nb_blocks_per_ring * self._nb_rings * np.prod(self._shape)
#
#     def fmap(self, f):
#         return Projection_meta(f(self._shape), f(self._center), f(self._size))
#
#     def transpose(self):
#         return self.fmap(lambda x: x[::-1])
#
#     def transfer(self, data_type = None):
#         raise NotImplementedError
#
#     def save_h5(self, path, mode = 'w'):
#         with h5py.File(path, mode) as fout:
#             group = fout.create_group('projection_meta')
#             group.attrs.create('_shape', data = self._shape)
#             group.attrs.create('_center', data = self._center)
#             group.attrs.create('_size', data = self._size)
#             group.attrs.create('_inner_radius', data = self._inner_radius)
#             group.attrs.create('_outer_radius', data = self._outer_radius)
#             group.attrs.create('_axial_length', data = self._axial_length)
#             group.attrs.create('_nb_rings', data = self._nb_rings)
#             group.attrs.create('_nb_blocks_per_ring', data = self._nb_blocks_per_ring)
#             group.attrs.create('_gap', data = self._gap)
#
#     def load_h5(path):
#         raise NotImplementedError
#
#
# class Listmode_projection_meta(Projection_meta):
#     def __init__(self, *argv):
#         super().__init__(*argv)
#
#     def transfer(self, data_type = None):
#         kwargs = self.__dict__
#         if data_type == 'sinogram':
#             return Sinogram_projection_meta(**kwargs)
#         elif data_type == 'listmode':
#             return self
#         else:
#             raise NotImplementedError
#
#     def load_h5(path):
#         with h5py.File(path, 'r') as fin:
#             group = fin['projection_meta']
#             _shape = tuple(group.attrs['_shape'])
#             _center = tuple(group.attrs['_center'])
#             _size = tuple(group.attrs['_size'])
#             _inner_radius = float(group.attrs['_inner_radius'])
#             _outer_radius = float(group.attrs['_outer_radius'])
#             _axial_length = float(group.attrs['_axial_length'])
#             _nb_rings = int(group.attrs['_nb_rings'])
#             _nb_blocks_per_ring = int(group.attrs['_nb_blocks_per_ring'])
#             _gap = float(group.attrs['_gap'])
#             return Listmode_projection_meta(_shape, _center, _size, _inner_radius, _outer_radius,
#                                             _axial_length,
#                                             _nb_rings, _nb_blocks_per_ring, _gap)
#
#
# class Sinogram_projection_meta(Projection_meta):
#     def __init__(self, *argv):
#         super().__init__(*argv)
#
#     def transfer(self, data_type = None):
#         kwargs = self.__dict__
#         if data_type == 'listmode':
#             return Listmode_projection_meta(**kwargs)
#         elif data_type == 'sinogram':
#             return self
#         else:
#             raise NotImplementedError
#
#     def load_h5(path):
#         with h5py.File(path, 'r') as fin:
#             group = fin['projection_meta']
#             _shape = tuple(group.attrs['_shape'])
#             _center = tuple(group.attrs['_center'])
#             _size = tuple(group.attrs['_size'])
#             _inner_radius = float(group.attrs['_inner_radius'])
#             _outer_radius = float(group.attrs['_outer_radius'])
#             _axial_length = float(group.attrs['_axial_length'])
#             _nb_rings = int(group.attrs['_nb_rings'])
#             _nb_blocks_per_ring = int(group.attrs['_nb_blocks_per_ring'])
#             _gap = float(group.attrs['_gap'])
#             return Sinogram_projection_meta(_shape, _center, _size, _inner_radius, _outer_radius,
#                                             _axial_length,
#                                             _nb_rings, _nb_blocks_per_ring, _gap)
#
#
# class Projection:
#     def __init__(self, meta: Projection_meta, _data = None):
#         self.meta = meta
#         self._data = _data
#
#     @property
#     def data(self):
#         return self._data
#
#     @property
#     def shape(self):
#         return self.meta._shape
#
#     @property
#     def center(self):
#         return self.meta._center
#
#     @property
#     def size(self):
#         return self.meta._size
#
#     @property
#     def unit_size(self):
#         return tuple(ti / tj for (ti, tj) in zip(self.meta._size, self.meta._shape))
#
#     @property
#     def inner_radius(self) -> float:
#         return self.meta._inner_radius
#
#     @property
#     def outer_radius(self) -> float:
#         return self.meta._outer_radius
#
#     @property
#     def axial_length(self) -> float:
#         return self.meta._axial_length
#
#     @property
#     def nb_rings(self):
#         return self.meta._nb_rings
#
#     @property
#     def nb_blocks_per_ring(self):
#         return self.meta._nb_blocks_per_ring
#
#     @property
#     def nb_gap(self):
#         return self.meta._gap
#
#     def fmap(self, f):
#         return Projection(self.meta, f(self._data))
#
#     def transpose(self):
#         return Projection(self.meta.transpose(), self._data.T)
#
#     def transfer(self):
#         raise NotImplementedError
#
#     def save_h5(self, path):
#         raise NotImplementedError
#
#     def load_h5(path):
#         raise NotImplementedError
#
#
# class Listmode_projection(Projection):
#     def __init__(self, meta: Listmode_projection_meta, data: np.ndarray):
#         super().__init__(meta, data)
#
#     @property
#     def fst(self):
#         return self.data[:, :3]
#
#     @property
#     def snd(self):
#         return self.data[:, 3:6]
#
#     @property
#     def vals(self):
#         return self.data[:, 6]
#
#     def save_h5(self, path):
#         self.meta.save_h5(path)
#         with h5py.File(path, 'r+') as fout:
#             group = fout.create_group('projection_data')
#             group.create_dataset('_data', data = self._data, compression = "gzip")
#
#     def load_h5(path):
#         meta = Listmode_projection_meta.load_h5(path)
#         with h5py.File(path, 'r') as fin:
#             group = fin['projection_data']
#             _data = np.array(group['_data'])
#             return Listmode_projection(meta, _data)
#
#     def transfer(self, data_type = None):
#         if data_type == 'sinogram':
#             data2 = _listmode_to_sinogram(self.meta, self._data)
#             return Sinogram_projection(self.meta.transfer('sinogram'), data2)
#         elif data_type == 'listmode':
#             return self
#         else:
#             raise NotImplementedError
#
#
# class Sinogram_projection(Projection):
#     def __init__(self, meta: Sinogram_projection_meta, data: sparse.csr_matrix):
#         super().__init__(meta, data)
#
#     @property
#     def fst(self):
#         return self.data.nonzero()[0]
#
#     @property
#     def snd(self):
#         return self.data.nonzero()[1]
#
#     @property
#     def vals(self):
#         return self.data.data
#
#     def save_h5(self, path):
#         self.meta.save_h5(path)
#         with h5py.File(path, 'r+') as fout:
#             group = fout.create_group('projection_data')
#             group.create_dataset('fst', data = self.fst, compression = "gzip")
#             group.create_dataset('snd', data = self.snd, compression = "gzip")
#             group.create_dataset('vals', data = self.vals, compression = "gzip")
#
#     def load_h5(path):
#         meta = Sinogram_projection_meta.load_h5(path)
#         with h5py.File(path, 'r') as fin:
#             group = fin['projection_data']
#             fst = np.array(group['fst'])
#             snd = np.array(group['snd'])
#             vals = np.array(group['vals'])
#             csr = sparse.csr_matrix((vals, (fst, snd)), shape = (meta.n_sensors, meta.n_sensors))
#             return Sinogram_projection(meta, csr)
#
#     def transfer(self, data_type = None):
#         if data_type == 'listmode':
#             data2 = _sinogram_to_listmode(self.meta, self.data)
#             return Listmode_projection(self.meta.transfer('listmode'), data2)
#         elif data_type == 'sinogram':
#             return self
#         else:
#             raise NotImplementedError
#
#
# def _listmode_to_sinogram(meta: Projection_meta, listmode: np.ndarray):
#     angle_per_block = np.pi * 2 / meta.nb_blocks_per_ring
#     x_data1 = listmode[:, 0]
#     y_data1 = listmode[:, 1]
#     z_data1 = listmode[:, 2]
#     angles1 = np.arctan2(y_data1, x_data1) + np.pi
#     iblock1 = (np.round(angles1 / angle_per_block) % meta.nb_blocks_per_ring).astype(int)
#     angle = iblock1 * angle_per_block
#     y_data1o = -x_data1 * np.sin(angle) + y_data1 * np.cos(angle)
#     y_ind1 = np.round(
#         (y_data1o + meta.size[1] / 2 - meta.unit_size[1] / 2) / meta.unit_size[1]).astype(int)
#     z_ind1 = np.round(
#         (z_data1 + meta.size[2] * meta.nb_rings / 2 - meta.unit_size[2] / 2) / meta.unit_size[
#             2]).astype(int)
#     iring1 = np.floor(z_ind1 / meta.shape[2]).astype(int)
#     z_ind1 //= meta.shape[2]
#
#     x_data2 = listmode[:, 3]
#     y_data2 = listmode[:, 4]
#     z_data2 = listmode[:, 5]
#     angles2 = np.arctan2(y_data2, x_data2) + np.pi
#     iblock2 = (np.round(angles2 / angle_per_block) % meta.nb_blocks_per_ring).astype(int)
#     angle = iblock2 * angle_per_block
#     y_data2o = -x_data2 * np.sin(angle) + y_data2 * np.cos(angle)
#     y_ind2 = np.round(
#         (y_data2o + meta.size[1] / 2 - meta.unit_size[1] / 2) / meta.unit_size[1]).astype(
#         int)
#     z_ind2 = np.round(
#         (z_data2 + meta.size[2] * meta.nb_rings / 2 - meta.unit_size[2] / 2) / meta.unit_size[
#             2]).astype(
#         int)
#     iring2 = np.floor(z_ind2 / meta.shape[2]).astype(int)
#     z_ind2 //= meta.shape[2]
#
#     n1 = meta.shape[1]
#     n2 = meta.shape[2] * n1
#     n3 = meta.nb_blocks_per_ring * n2
#     n4 = meta.nb_rings * n3
#     ind1 = (y_ind1 + n1 * z_ind1 + n2 * iblock1 + n3 * iring1).astype(int)
#     ind2 = (y_ind2 + n1 * z_ind2 + n2 * iblock2 + n3 * iring2).astype(int)
#     lil_mat = sparse.lil_matrix((n4, n4), dtype = np.float32)
#     for i in range(ind1.size):
#         lil_mat[ind1[i], ind2[i]] += listmode[i, 6]
#         lil_mat[ind2[i], ind1[i]] += listmode[i, 6]
#     return lil_mat.tocsr()
#
#
# def _sinogram_to_listmode(meta: Projection_meta, sino: sparse.csr_matrix):
#     rows0, cols0 = sino.nonzero()
#     rows, cols = rows0[rows0 > cols0], cols0[rows0 > cols0]
#     vals = sino.data[rows0 > cols0]
#
#     x1_0 = x2_0 = (meta.inner_radius + meta.outer_radius) / 2
#     y_ind1 = (rows % meta.shape[1]).astype(int)
#     y1_0 = (y_ind1 + 0.5) * meta.unit_size[1] - meta.size[1] / 2
#     z_ind1 = (rows // meta.shape[1] % meta.shape[2]).astype(int)
#     iblock1 = (rows // meta.shape[1] // meta.shape[2] % meta.nb_blocks_per_ring).astype(int)
#     iring1 = (rows // meta.shape[1] // meta.shape[2] // meta.nb_blocks_per_ring).astype(int)
#
#     y_ind2 = (cols % meta.shape[1]).astype(int)
#     y2_0 = (y_ind2 + 0.5) * meta.unit_size[1] - meta.size[1] / 2
#     z_ind2 = (cols // meta.shape[1] % meta.shape[2]).astype(int)
#     iblock2 = (cols // meta.shape[1] // meta.shape[2] % meta.nb_blocks_per_ring).astype(int)
#     iring2 = (cols // meta.shape[1] // meta.shape[2] // meta.nb_blocks_per_ring).astype(int)
#     angle_per_block = np.pi * 2 / meta.nb_blocks_per_ring
#
#     data2 = np.zeros((vals.size, 7))
#
#     data2[:, 0] = x1_0 * np.cos(iblock1 * angle_per_block) - y1_0 * np.sin(
#         iblock1 * angle_per_block)
#     data2[:, 1] = x1_0 * np.sin(iblock1 * angle_per_block) + y1_0 * np.cos(
#         iblock1 * angle_per_block)
#     data2[:, 2] = (0.5 + z_ind1) * meta.unit_size[2] + iring1 * meta.size[2] - meta.nb_rings * \
#                   meta.size[2] / 2
#
#     data2[:, 3] = x2_0 * np.cos(iblock2 * angle_per_block) - y2_0 * np.sin(
#         iblock2 * angle_per_block)
#     data2[:, 4] = x2_0 * np.sin(iblock2 * angle_per_block) + y2_0 * np.cos(
#         iblock2 * angle_per_block)
#     data2[:, 5] = (0.5 + z_ind2) * meta.unit_size[2] + iring2 * meta.size[2] - meta.nb_rings * \
#                   meta.size[2] / 2
#
#     data2[:, 6] = vals
#     return data2
