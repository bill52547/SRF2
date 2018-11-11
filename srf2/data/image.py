import h5py
import numpy as np

from srf2.meta.image_meta import *

__all__ = ('Image', 'Image_2d', 'Image_3d',)


class Image:
    def __init__(self, data = None, meta: Image_meta = None):
        if meta is None and data is None:
            meta = Image_meta()
            data = np.zeros(meta.shape, dtype = np.float32)

        if data is None:
            data = np.zeros(meta.shape, dtype = np.float32)

        if meta is None:
            meta = Image_meta(data.shape)

        if not isinstance(data, np.ndarray):
            raise TypeError('data must be a np.ndarray object')

        if data.shape != meta.shape:
            raise ValueError('data must has same shape with its meta')

        data.astype(np.float32)

        self._data = data
        self._meta = meta

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False

        return self.meta == other.meta and np.array_equal(self.data, other.data)

    @property
    def data(self):
        return self._data

    @property
    def meta(self):
        return self._meta

    @property
    def T(self):
        return self.transpose()

    def fmap(self, f):
        return Image(f(self.data), self.meta)

    def transpose(self, perm = None):
        if perm is None:
            perm = np.arange(self.meta.ndim)[::-1]

        if set(perm).issubset({'x', 'y', 'z'}):
            perm = [self.meta.dims.index(e) for e in perm]

        return Image(self.data.transpose(perm), self.meta.transpose(perm))

    # TODO slicing operator, accoridng to image_meta

    def save_h5(self, path = None, mode = 'w'):
        if path is None:
            path = 'tmp' + self.__class__.__name__ + '.h5'
        self.meta.save_h5(path, mode)
        with h5py.File(path, 'r+') as fout:
            group = fout.create_group(self.__class__.__name__)
            group.create_dataset('_data', data = self.data, compression = "gzip")

    @classmethod
    def load_h5(cls, path = None):
        if path is None:
            path = 'tmp' + cls.__name__ + '.h5'
        meta = Image_meta.load_h5(path)
        with h5py.File(path, 'r') as fin:
            dataset = fin[cls.__name__]
            data = np.array(dataset['_data'])
            return cls(data, meta)


class Image_2d(Image):
    def __init__(self, data = None, meta: Image_meta_2d = None):
        super().__init__(data, meta)
        if meta.ndim != 2:
            raise ValueError(self.__class__.__name__, ' is only consistent with 2D case')


class Image_3d(Image):
    def __init__(self, data = None, meta: Image_meta_3d = None):
        super().__init__(data, meta)
        if meta.ndim != 3:
            raise ValueError(self.__class__.__name__, ' is only consistent with 2D case')
