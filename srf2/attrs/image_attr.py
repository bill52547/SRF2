import numpy as np
from functools import reduce
from abc import abstractmethod

from srf2.core.abstracts import Attribute

__all__ = ('Image_1d_attr', 'Image_2d_attr', 'Image_3d_attr', 'Image_4d_attr',)


class Image_attr(Attribute):
    def __init__(self, shape=None, center=None, size=None, dims=None):
        self._shape = tuple(shape) if shape is not None else (1, 1, 1)
        self._center = tuple(center) if center is not None else (0, 0, 0)
        self._size = tuple(size) if size is not None else (1, 1, 1)
        self._dims = tuple(dims) if dims is not None else ('x', 'y', 'z')
        if not (len(self._shape) == len(self._center) == len(self._size) == len(self._dims)):
            raise ValueError(self.__dict__, ' should have same lengths')
        if len(self._shape) < 1 or len(self._shape) > 3:
            raise NotImplemented

    def _dim_to_int(self, dim: str = None):
        if dim is None:
            return slice(None)
        else:
            return [self._dims.index(s) for s in dim]

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
    def T(self):
        return self.transpose()

    def num(self, dim=None):
        if dim is str:
            dim = [s for s in dim]
        if set(dim).issubset(self.dims):
            dim = [self.dims.index(e) for e in dim]

        nums = self.shape[dim]
        return reduce(lambda x, y: x * y, nums)

    @abstractmethod
    def map(self, f):
        pass

    @abstractmethod
    def meshgrid(self):
        pass

    @abstractmethod
    def unit_centers(self):
        pass

    def __getitem__(self, ind):
        shape = list(self.shape)
        center = list(self.center)
        size = list(self.size)
        dims = self.dims

        for k in range(self.ndim):
            if not isinstance(ind[k], (slice, int)):
                raise TypeError('list indices must be integers or slices, not ', type(ind[k]))

            if ind[k] is int:
                if not -self.ndim < ind[k] < self.ndim:
                    raise IndexError(self.__class__.__name__, ' index out of range')
                shape[k] = 1
                center[k] = center[k][ind[k]]
                size[k] = size[k][ind[k]]
            else:
                if ind[k].step is None:
                    step = 1
                else:
                    step = ind[k].step
                rang = range(shape[k])[ind[k]]
                unit_size = size[k] / shape[k]
                center[k] = (rang[0] + rang[-1] + 1) / 2 * unit_size + center[k] - size[k] / 2
                shape[k] = len(rang)
                size[k] = shape[k] * unit_size * step

        count_nonzero = shape.count(1)
        if count_nonzero == 1:
            return Image_1d_attr(shape, center, size, dims)
        elif count_nonzero == 2:
            return Image_2d_attr(shape, center, size, dims)
        elif count_nonzero == 3:
            return Image_3d_attr(shape, center, size, dims)
        else:
            raise NotImplementedError

    def locate(self, pos=None):
        if pos is None:
            return ValueError('No valid input.')
        result = [0] * self.ndim
        for k in range(self.ndim):
            result[k] = (pos[k] - self.center[k] + self.size[k] / 2) / self.unit_size[k] - 0.5
        return tuple(result)

    def transpose(self, perm=None):
        if perm is None:
            perm = np.arange(self.ndim)[::-1]
        if set(perm).issubset({'x', 'y', 'z'}):
            perm = [self.dims.index(e) for e in perm]
        shape = [self.shape[i] for i in perm]
        center = [self.center[i] for i in perm]
        size = [self.size[i] for i in perm]
        dims = [self.dims[i] for i in perm]
        return self.__class__(shape, center, size, dims)


class Image_1d_attr(Image_attr):
    def __init__(self, shape=(1,), center=(0,), size=(1,), dims=('x',)):
        super().__init__(shape, center, size, dims)
        if len(shape) != 1:
            raise ValueError(self.__class__, ' is only consistent with 2D case')

    def map(self, f):
        return Image_1d_attr(*f(self.shape, self.center, self.size, self.dims))

    def meshgrid(self):
        return np.arange(self.shape[0])

    def unit_centers(self):
        return self.meshgrid() * self.unit_size[0] + self.center[0] - self.size[0] / 2 + self.unit_size[0] / 2


class Image_2d_attr(Image_attr):
    def __init__(self, shape=(1, 1), center=(0, 0), size=(1, 1), dims=('x', 'y')):
        super().__init__(shape, center, size, dims)
        if len(shape) != 2:
            raise ValueError(self.__class__, ' is only consistent with 2D case')

    def map(self, f):
        return Image_2d_attr(*f(self.shape, self.center, self.size, self.dims))

    def meshgrid(self, slice=None):
        x = np.arange(self.shape[0])
        y = np.arange(self.shape[1])
        y1, x1 = np.meshgrid(y, x)
        return x1, y1

    def unit_centers(self):
        x1, y1 = self.meshgrid(slice)
        pos_x = x1 * self.unit_size[0] + self.center[0] - self.size[0] / 2 + self.unit_size[0] / 2
        pos_y = y1 * self.unit_size[1] + self.center[1] - self.size[1] / 2 + self.unit_size[1] / 2
        return pos_x, pos_y

    def theta(self):
        x1, y1 = self.unit_centers()
        return np.arctan2(y1, x1)


class Image_3d_attr(Image_attr):
    def __init__(self, shape=(1, 1, 1), center=(0, 0, 0), size=(1, 1, 1),
                 dims=('x', 'y', 'z')):
        super().__init__(shape, center, size, dims)
        if len(shape) != 3:
            raise ValueError(self.__class__, ' is only consistent with 2D case')

    def map(self, f):
        return Image_3d_attr(*f(self.shape, self.center, self.size, self.dims))

    def meshgrid(self):
        x = np.arange(self.shape[0])
        y = np.arange(self.shape[1])
        z = np.arange(self.shape[2])
        (y1, x1, z1) = np.meshgrid(y, x, z)
        return x1, y1, z1

    def unit_centers(self):
        x1, y1, z1 = self.meshgrid()
        pos_x = x1 * self.unit_size[0] + self.center[0] - self.size[0] / 2 + self.unit_size[0] / 2
        pos_y = y1 * self.unit_size[1] + self.center[1] - self.size[1] / 2 + self.unit_size[1] / 2
        pos_z = z1 * self.unit_size[2] + self.center[2] - self.size[2] / 2 + self.unit_size[2] / 2
        return pos_x, pos_y, pos_z

