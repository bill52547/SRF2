import numpy as np
from ..attrs.imageattr import *
from ..core.abstracts import Object

__all__ = ('Image', 'Image0D', 'Image1D', 'Image2D', 'Image3D', 'Image4D',)


class Image(Object):
    _data: np.ndarray
    _attr: ImageAttr

    def __init__(self, data=None, attr: ImageAttr = None):
        if attr is None and data is None:
            attr = ImageAttr()
            data = np.zeros(attr.shape, dtype=np.float32)

        if data is None:
            data = np.zeros(attr.shape, dtype=np.float32)

        if attr is None:
            attr = ImageAttr(data.shape)

        if not isinstance(data, np.ndarray):
            raise TypeError('data must be a np.ndarray object')

        if data.shape != attr.shape:
            raise ValueError('data must has same shape with its attrs')

        data.astype(np.float32)
        self._data = data
        self._attr = attr

    @property
    def data(self):
        return self._data

    @property
    def attr(self):
        return self._attr

    @property
    def T(self):
        return self.transpose()

    def normalize(self):
        def _normalize(data):
            return data / np.sum(data)

        return self.map(_normalize)

    def transpose(self, perm=None):
        if perm is None:
            perm = np.arange(self.attr.ndim)[::-1]
        if set(perm).issubset({'x', 'y', 'z'}):
            perm = [self.attr.dims.index(e) for e in perm]

        def _transpose(data, attr):
            return data.transpose(perm), attr.transpose(attr)

        return self.map(_transpose)

    def map(self, f):
        return self.__class__(*f(self.data, self.attr))

    def __getitem__(self, item):
        attr = self.attr[item]
        data = self.data[item]
        count_nonzero = attr.shape.count(1)
        if count_nonzero == 1:
            return Image1D(data, attr)
        elif count_nonzero == 2:
            return Image2D(data, attr)
        elif count_nonzero == 3:
            return Image3D(data, attr)
        else:
            raise NotImplementedError


class Image0D(Image):
    def __init__(self, data=None, attr: Image1DAttr = None):
        super().__init__(data, attr)
        if self.attr.ndim != 0:
            raise ValueError(self.__class__.__name__, ' is only consistent with 0D case')


class Image1D(Image):
    def __init__(self, data=None, attr: Image1DAttr = None):
        super().__init__(data, attr)
        if self.attr.ndim != 1:
            raise ValueError(self.__class__.__name__, ' is only consistent with 1D case')


class Image2D(Image):
    def __init__(self, data=None, attr: Image2DAttr = None):
        if self.attr.ndim != 2:
            raise ValueError(self.__class__.__name__, ' is only consistent with 2D case')
        super().__init__(data, attr)


class Image3D(Image):
    def __init__(self, data=None, attr: Image3DAttr = None):
        super().__init__(data, attr)
        if self.attr.ndim != 3:
            raise ValueError(self.__class__.__name__, ' is only consistent with 3D case')


class Image4D(Image):
    def __init__(self, data=None, attr: Image4DAttr = None):
        super().__init__(data, attr)
        if self.attr.ndim != 4:
            raise ValueError(self.__class__.__name__, ' is only consistent with 2D case')
