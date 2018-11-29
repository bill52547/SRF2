import numpy as np

from ..attr.image_attr import *
from ..core.abstracts import Object

__all__ = ('Image', 'Image0D', 'Image1D', 'Image2D', 'Image3D',)


class Image(Object):
    def __init__(self, attr: ImageAttr, data = None):
        if attr is None:
            raise ValueError
        self._data = data
        self._attr = attr
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise ValueError
            self._data.astype(np.float32)

    def __bool__(self):
        return False if self.data is None else True

    def map(self, f):
        return self.__class__(*f(self.attr, self.data))

    def transpose(self, perm = None):
        if perm is None:
            perm = np.arange(self.attr.ndim)[::-1]
        if set(perm).issubset({'x', 'y', 'z'}):
            perm = [self.attr.dims.index(e) for e in perm]

        def _transpose(attr, data):
            if data is None:
                return attr.transpose(perm), None
            else:
                return attr.transpose(perm), data.transpose(perm)

        return self.map(_transpose)

    @property
    def T(self):
        return self.transpose()

    def __getitem__(self, item):
        attr = self.attr[item]
        data = self.data[item]
        return self.__class__(attr, data)

    def squeeze(self):
        attr = self.attr.squeeze()
        data = self.data.squeeze()
        if isinstance(attr, Image0DAttr):
            return Image0D(attr, data)
        elif isinstance(attr, Image1DAttr):
            return Image1D(attr, data)
        elif isinstance(attr, Image2DAttr):
            return Image2D(attr, data)
        elif isinstance(attr, Image3DAttr):
            return Image3D(attr, data)
        else:
            raise NotImplementedError


class Image0D(Image):
    def __init__(self, attr: Image0DAttr = None, data = None):
        super().__init__(attr, data)
        if self.attr.ndim != 0:
            raise ValueError(self.__class__.__name__, ' is only consistent with 0D case')


class Image1D(Image):
    def __init__(self, attr: Image1DAttr = None, data = None):
        super().__init__(attr, data)
        if self.attr.ndim != 1:
            raise ValueError(self.__class__.__name__, ' is only consistent with 1D case')


class Image2D(Image):
    def __init__(self, attr: Image2DAttr = None, data = None):
        super().__init__(attr, data)
        if self.attr.ndim != 2:
            raise ValueError(self.__class__.__name__, ' is only consistent with 2D case')


class Image3D(Image):
    def __init__(self, attr: Image3DAttr = None, data = None):
        super().__init__(attr, data)
        if self.attr.ndim != 3:
            raise ValueError(self.__class__.__name__, ' is only consistent with 3D case')
