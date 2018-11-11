import attr
import numpy as np

from srf2.core.abstracts import Meta, Singleton

__all__ = ('Image_meta', 'Image_meta_singleton', 'Image_meta_2d', 'Image_meta_2d_singleton',
           'Image_meta_3d', 'Image_meta_3d_singleton',)

@attr.s
class Image_meta(Meta):
    _shape = attr.ib(default = (1,) * 3)
    _center = attr.ib(default = (0,) * 3)
    _size = attr.ib(default = (1,) * 3)
    _dims = attr.ib(default = ('x', 'y', 'z'))

    @_shape.validator
    def _check_arg_shape(self, attribute, value):
        if not 2 <= len(value) <= 4:
            raise ValueError('_shape must in range [2, 4]')
        if len(value) == 4:
            raise NotImplementedError
        if not isinstance(value, tuple):
            raise TypeError('_shape must be a tuple')

    @_center.validator
    def _check_arg_center(self, attribute, value):
        if len(value) != len(self._shape):
            raise ValueError('_center must have same size with _shape')
        if not isinstance(value, tuple):
            raise TypeError('_center must be a tuple')

    @_size.validator
    def _check_arg_size(self, attribute, value):
        if len(value) != len(self._shape):
            raise ValueError('_size must have same size with _shape')
        if not isinstance(value, tuple):
            raise TypeError('_size must be a tuple')

    @_dims.validator
    def _check_arg_dims(self, attribute, value):
        if len(value) != len(self._shape):
            raise ValueError('_dims must have same size with _shape')
        if not isinstance(value, tuple):
            raise TypeError('_dims must be a tuple')
        if not set(value).issubset({'x', 'y', 'z', 't'}):
            return ValueError('Only x, y, z and t are allowed int _dims')
        if 't' in value:
            raise NotImplementedError

    @property
    def shape(self):
        return self._shape

    @property
    def center(self):
        return self._center

    @property
    def size(self):
        return self._size

    @property
    def dims(self):
        return self._dims

    @property
    def unit_size(self):
        return tuple(ti / tj for (ti, tj) in zip(self._size, self._shape))

    @property
    def ndim(self):
        return len(self._dims)

    @property
    def n_x(self):
        return self._shape[self._dims.index('x')] if 'x' in self._dims else 1

    @property
    def n_y(self):
        return self._shape[self._dims.index('y')] if 'y' in self._dims else 1

    @property
    def n_z(self):
        return self._shape[self._dims.index('z')] if 'z' in self._dims else 1

    @property
    def n_t(self):
        return self._shape[self._dims.index('t')] if 't' in self._dims else 1

    @property
    def n_all(self):
        return np.array(self.shape).prod()

    def slice(self, start = None, stop = None, step = 1):
        ind = slice(start, stop, step)
        return Image_meta(self.shape[ind], self.center[ind], self.size[ind], self.dims[ind])

    def transpose(self, perm = None):
        if not perm:
            perm = range(self.ndim)[::-1]
        if 't' in perm:
            raise NotImplementedError

        if set(perm).issubset({'x', 'y', 'z'}):
            perm = [self.dims.index(e) for e in perm]

        shape = tuple([self.shape[i] for i in perm])
        center = tuple([self.center[i] for i in perm])
        size = tuple([self.size[i] for i in perm])
        dims = tuple([self.dims[i] for i in perm])
        return Image_meta(shape, center, size, dims)

    def fmap(self, f):
        raise NotImplementedError


class Image_meta_singleton(Image_meta, Singleton):
    pass
    # instance = None
    #
    # def __new__(cls, *args, **kwargs):
    #     if cls.instance is None:
    #         cls.instance = super().__new__(cls)
    #     return cls.instance


class Image_meta_2d(Image_meta):
    def __init__(self, shape = (1, 1), center = (0, 0), size = (1, 1), dims = ('x', 'y')):
        if len(shape) != 2:
            raise ValueError(self.__class__, ' is only consistent with 2D case')
        super().__init__(shape, center, size, dims)

    def meshgrid(self):
        x = np.arange(self.shape[0]) * self.unit_size[0] + self.center[0] - self.size[0] / 2 + \
            self.unit_size[0] / 2
        y = np.arange(self.shape[1]) * self.unit_size[1] + self.center[1] - self.size[1] / 2 + \
            self.unit_size[1] / 2
        y1, x1 = np.meshgrid(y, x)
        return x1, y1

    def theta(self):
        x1, y1 = self.meshgrid()
        return np.arctan2(y1, x1)


class Image_meta_2d_singleton(Image_meta_2d, Singleton):
    pass


class Image_meta_3d(Image_meta):
    def __init__(self, shape = (1, 1, 1), center = (0, 0, 0), size = (1, 1, 1),
                 dims = ('x', 'y', 'z')):
        if len(shape) != 3:
            raise ValueError(self.__class__, ' is only consistent with 2D case')
        super().__init__(shape, center, size, dims)

    def meshgrid(self):
        x = np.arange(self.shape[0]) * self.unit_size[0] + self.center[0] - self.size[0] / 2 + \
            self.unit_size[0] / 2
        y = np.arange(self.shape[1]) * self.unit_size[1] + self.center[1] - self.size[1] / 2 + \
            self.unit_size[1] / 2
        z = np.arange(self.shape[2]) * self.unit_size[2] + self.center[2] - self.size[2] / 2 + \
            self.unit_size[2] / 2

        (y1, x1, z1) = np.meshgrid(y, x, z)
        return x1, y1, z1


class Image_meta_3d_singleton(Image_meta_3d, Singleton):
    pass
