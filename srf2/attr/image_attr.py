from abc import abstractmethod
from functools import reduce

import numpy as np

from srf2.core.abstracts import AttributeWithShape

__all__ = ('ImageAttr', 'Image0DAttr', 'Image1DAttr', 'Image2DAttr', 'Image3DAttr', 'Image4DAttr',)


class ImageAttr(AttributeWithShape):
    def __init__(self, shape=None, center=None, size=None, dims=None):
        super().__init__(shape)
        self._center = tuple(center) if center is not None else tuple([0 for _ in shape])
        self._size = tuple(size) if size is not None else tuple([k for k in shape])
        self._dims = tuple(dims) if dims is not None else ('x', 'y', 'z', 't')[:len(shape)]
        if not (len(self._shape) == len(self._center) == len(self._size) == len(shape)):
            raise ValueError(self.__dict__, ' should have same lengths')
        if len(shape) < 0 or len(shape) > 4:
            raise NotImplemented

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
        if 'x' in self.dims:
            return self.shape[self.dims.index('x')]
        else:
            return 1

    @property
    def n_y(self):
        if 'y' in self.dims:
            return self.shape[self.dims.index('y')]
        else:
            return 1

    @property
    def n_z(self):
        if 'z' in self.dims:
            return self.shape[self.dims.index('z')]
        else:
            return 1

    @property
    def n_t(self):
        if 't' in self.dims:
            return self.shape[self.dims.index('t')]
        else:
            return 1

    def numel(self, dims=None):
        if dims is None:
            dims = self.dims
        if dims is str:
            dims = [s for s in dims]
        if set(dims).issubset(self.dims):
            dims = [self.dims.index(e) for e in dims]
        return reduce(lambda x, y: x * y, [self.shape[s] for s in dims])

    def map(self, f):
        return self.__class__(*f(self.shape, self.center, self.size, self.dims))

    @abstractmethod
    def meshgrid(self):
        raise NotImplementedError

    @abstractmethod
    def unit_centers(self):
        raise NotImplementedError

    def linspace(self, item=None):
        if item is None:
            return None
        if isinstance(item, str):
            item = [s for s in item]
        if isinstance(item, (list, tuple)):
            return [self.linspace(it) for it in item]
        if set(item).issubset({'x', 'y', 'z', 't'}):
            item = self.dims.index(item)
        return (np.arange(self.shape[item]) - self.shape[item] / 2 + 0.5) * self.unit_size[item] + \
               self.center[item]

    def __getitem__(self, item):
        shape = list(self.shape)
        center = list(self.center)
        size = list(self.size)
        dims = self.dims

        for k in range(self.ndim):
            if isinstance(item, (slice, int)):
                item = (item,)
            if not isinstance(item[k], (slice, int)):
                raise TypeError('list indices must be integers or slices, not ', type(item[k]))

            if item[k] is int:
                if not -self.ndim < item[k] < self.ndim:
                    raise IndexError(self.__class__.__name__, ' index out of range')
                shape[k] = 1
                center[k] = center[k][item[k]]
                size[k] = size[k][item[k]]
            else:
                if item[k].step is None:
                    step = 1
                else:
                    step = item[k].step
                rang = range(shape[k])[item[k]]
                unit_size = size[k] / shape[k]
                center[k] = (rang[0] + rang[-1] + 1) / 2 * unit_size + center[k] - size[k] / 2
                shape[k] = len(rang)
                size[k] = shape[k] * unit_size * step
        return self.__class__(shape, center, size, dims)

    def squeeze(self):
        index_nonzero = [i for i in range(self.ndim) if self.shape[i] > 1]
        shape = [self.shape[e] for e in index_nonzero]
        center = [self.center[e] for e in index_nonzero]
        size = [self.size[e] for e in index_nonzero]
        dims = [self.dims[e] for e in index_nonzero]
        count_nonzero = len(index_nonzero)
        if count_nonzero == 0:
            return Image0DAttr(shape, center, size, dims)
        elif count_nonzero == 1:
            return Image1DAttr(shape, center, size, dims)
        elif count_nonzero == 2:
            return Image2DAttr(shape, center, size, dims)
        elif count_nonzero == 3:
            return Image3DAttr(shape, center, size, dims)
        else:
            raise NotImplementedError

    def index(self, pos=None):
        if pos is None:
            return ValueError('No valid input.')
        if np.isscalar(pos):
            pos = (pos,)
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

    @property
    def T(self):
        return self.transpose()


class Image0DAttr(ImageAttr):
    def __init__(self, shape=None, center=None, size=None, dims=None):
        if shape is None:
            shape = tuple([])
        super().__init__(shape, center, size, dims)
        if len(self.shape) != 0:
            raise ValueError(self.__class__, ' is only consistent with 0D case')

    def meshgrid(self):
        raise ValueError

    def unit_centers(self):
        raise ValueError


class Image1DAttr(ImageAttr):
    def __init__(self, shape=None, center=None, size=None, dims=None):
        if shape is None:
            shape = (1,)

        super().__init__(shape, center, size, dims)
        if len(self.shape) != 1:
            raise ValueError(self.__class__, ' is only consistent with 1D case')

    def meshgrid(self):
        return np.arange(self.shape[0])

    def unit_centers(self):
        return self.linspace(0)


class Image2DAttr(ImageAttr):
    def __init__(self, shape=None, center=None, size=None, dims=None):
        if shape is None:
            shape = (1, 1)
        super().__init__(shape, center, size, dims)
        if len(self.shape) != 2:
            raise ValueError(self.__class__, ' is only consistent with 2D case')

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


class Image3DAttr(ImageAttr):
    def __init__(self, shape=None, center=None, size=None, dims=None):
        if shape is None:
            shape = (1, 1, 1)
        super().__init__(shape, center, size, dims)
        if len(self.shape) != 3:
            raise ValueError(self.__class__, ' is only consistent with 3D case')

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


class Image4DAttr(ImageAttr):
    def __init__(self, shape=None, center=None, size=None, dims=None):
        if shape is None:
            shape = (1, 1, 1, 1)
        super().__init__(shape, center, size, dims)
        if len(self.shape) != 4:
            raise ValueError(self.__class__, ' is only consistent with 4D case')

    def meshgrid(self):
        raise NotImplementedError

    def unit_centers(self):
        raise NotImplementedError
