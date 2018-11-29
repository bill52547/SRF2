#!/usr/bin/env python
# encoding: utf-8
'''
@author: Minghao Guo, Xiang Hong, Gaoyu Chen and Weijie Tao
@license: LGPL_v3.0
@contact: mh.guo0111@gmail.com
@software: srf_v2
@file: projection_attr.py
@date: 11/11/2018
@desc: new version of Scalable Reconstruction Framework for Medical Imaging
'''

from abc import abstractmethod

import numpy as np

from srf2.core.abstracts import Attribute

__all__ = ('DetectorAttr', 'Detector1DAttr', 'Detector2DAttr', 'ProjectionAttr',
           'ProjectionFlatAttr', 'ProjectionCurveAttr',)


class DetectorAttr(Attribute):
    def __init__(self, shape = None, center = None, size = None, dims = None):
        self._shape = tuple(shape) if shape is not None else tuple([])
        self._center = tuple(center) if center is not None else tuple([0 for _ in self._shape])
        self._size = tuple(size) if size is not None else tuple([k for k in self._shape])
        self._dims = tuple(dims) if dims is not None else ('u', 'v')[:len(self._shape)]

        if not (len(self._shape) == len(self._center) == len(self._size) == len(self._dims)):
            raise ValueError(self.__dict__, ' should have same lengths')
        if len(self._shape) < 0 or len(self._shape) > 2:
            raise NotImplemented

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
        return tuple([x / y for (x, y) in zip(self.size, self.shape)])

    @property
    def n_u(self):
        if 'u' in self.dims:
            return self.shape[self.dims.index('u')]
        else:
            return 1

    @property
    def n_v(self):
        if 'v' in self.dims:
            return self.shape[self.dims.index('v')]
        else:
            return 1

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def numel(self):
        return self.n_u * self.n_v

    @abstractmethod
    def meshgrid(self):
        pass

    @abstractmethod
    def unit_centers(self):
        pass

    def map(self, f):
        return self.__class__(*f(self.shape, self.center, self.size, self.dims))

    def __getitem__(self, item):
        shape = list(self.shape)
        center = list(self.center)
        size = list(self.size)
        dims = list(self.dims)
        for k in range(self.ndim):
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

    def locate(self, pos = None):
        if pos is None:
            return ValueError('No valid input.')
        if np.isscalar(pos):
            pos = (pos,)
        result = [0] * self.ndim
        if self.dims == ('v', 'u',) or self.dims == ('v',):
            raise NotImplementedError
        for k in range(self.ndim):
            result[k] = (pos[k] - self.center[k] + self.size[k] / 2) / self.unit_size[k] - 0.5
        return tuple(result)

    def transpose(self, perm = None):
        if perm is None:
            perm = np.arange(self.ndim)[::-1]
        if set(perm).issubset({'u', 'v'}):
            perm = [self.dims.index(e) for e in perm]
        shape = [self.shape[i] for i in perm]
        center = [self.center[i] for i in perm]
        size = [self.size[i] for i in perm]
        dims = [self.dims[i] for i in perm]
        return self.__class__(shape, center, size, dims)

    @property
    def T(self):
        return self.transpose()

    def squeeze(self):
        index_nonzero = [i for i in range(self.ndim) if self.shape[i] > 1]
        shape = [self.shape[e] for e in index_nonzero]
        center = [self.center[e] for e in index_nonzero]
        size = [self.size[e] for e in index_nonzero]
        dims = [self.dims[e] for e in index_nonzero]
        count_nonzero = len(index_nonzero)
        if count_nonzero == 0:
            return Detector0DAttr(shape, center, size, dims)
        elif count_nonzero == 1:
            return Detector1DAttr(shape, center, size, dims)
        elif count_nonzero == 2:
            return Detector2DAttr(shape, center, size, dims)
        else:
            raise NotImplementedError


class Detector0DAttr(DetectorAttr):
    def __init__(self, shape = None, center = None, size = None, dims = None):
        if shape is None:
            shape = tuple([])
        super().__init__(shape, center, size, dims)
        if self.ndim != 0:
            raise ValueError(self.__class__, ' is only consistent with 0D case')

    def meshgrid(self):
        raise NotImplementedError

    def unit_centers(self):
        raise NotImplementedError


class Detector1DAttr(DetectorAttr):
    def __init__(self, shape = None, center = None, size = None, dims = None):
        if shape is None:
            shape = (1,)
        super().__init__(shape, center, size, dims)
        if self.ndim != 1:
            raise ValueError(self.__class__, ' is only consistent with 1D case')

    def meshgrid(self):
        return np.arange(self.numel)

    def unit_centers(self):
        if self.dims == ('v', 'u',):
            raise NotImplementedError

        return self.meshgrid() * self.unit_size[0] + self.center[0] - self.size[0] / 2 + \
               self.unit_size[0] / 2


class Detector2DAttr(DetectorAttr):
    def __init__(self, shape = None, center = None, size = None, dims = None):
        if shape is None:
            shape = (1, 1)
        super().__init__(shape, center, size, dims)
        if self.ndim != 2:
            raise ValueError(self.__class__, ' is only consistent with 2D case')

    def meshgrid(self):
        x = np.arange(self.shape[0])
        y = np.arange(self.shape[1])
        y1, x1 = np.meshgrid(y, x)
        return x1, y1

    def unit_centers(self):
        if self.dims == ('v', 'u',):
            raise NotImplementedError

        x1, y1 = self.meshgrid()
        pos_x = x1 * self.unit_size[0] + self.center[0] - self.size[0] / 2 + self.unit_size[0] / 2
        pos_y = y1 * self.unit_size[1] + self.center[1] - self.size[1] / 2 + self.unit_size[1] / 2
        return pos_x, pos_y


class ProjectionAttr(Attribute):
    def __init__(self, source_to_detector, source_to_image, angle, detector_attr: DetectorAttr):
        if source_to_detector < source_to_image:
            raise ValueError
        self._source_to_detector = np.float32(source_to_detector)
        self._source_to_image = np.float32(source_to_image)
        self._angle = np.float32(angle)
        if detector_attr.ndim == 0:
            self._detector_attr = Detector0DAttr(*detector_attr.__dict__.values())
        elif detector_attr.ndim == 1:
            self._detector_attr = Detector1DAttr(*detector_attr.__dict__.values())
        elif detector_attr.ndim == 2:
            self._detector_attr = Detector2DAttr(*detector_attr.__dict__.values())
        else:
            raise NotImplementedError

    @property
    def source_to_detector(self):
        return self._source_to_detector

    @property
    def source_to_image(self):
        return self._source_to_image

    @property
    def angle(self):
        return self._angle

    @property
    def detector_attr(self):
        return self._detector_attr

    @property
    def source_positions(self):
        x, y = -self.source_to_image * np.cos(self.angle), \
               -self.source_to_image * np.sin(self.angle)
        if isinstance(self.detector_attr, Detector1DAttr):
            return (x, y)
        elif isinstance(self.detector_attr, Detector2DAttr):
            return (x, y, 0)
        else:
            raise NotImplementedError

    @abstractmethod
    def detector_unit_centers(self):
        pass

    @abstractmethod
    def locate(self, pos):
        pass

    def map(self, f):
        return self.__class__(
            *f(self.source_to_detector, self.source_to_image, self.angle, self.detector_attr)
        )

    def squeeze(self):
        detector_attr = self.detector_attr.squeeze()
        return ProjectionAttr(self.source_to_detector, self.source_to_image, self.angle,
                              detector_attr)


class ProjectionFlatAttr(ProjectionAttr):
    def detector_unit_centers(self):
        if isinstance(self.detector_attr, Detector1DAttr):
            u = self.detector_attr.unit_centers()
            x = np.cos(self.angle) * (self.source_to_detector - self.source_to_image) \
                - np.sin(self.angle) * u
            y = np.cos(self.angle) * u
            return x, y
        elif isinstance(self.detector_attr, Detector2DAttr):
            u, v = self.detector_attr.unit_centers()
            x = np.cos(self.angle) * (self.source_to_detector - self.source_to_image) \
                - np.sin(self.angle) * u
            y = np.cos(self.angle) * u
            z = v
            return x, y, z
        else:
            raise NotImplementedError

    def locate(self, pos):
        if isinstance(self.detector_attr, Detector1DAttr):
            if len(pos) > 2:
                raise ValueError
            x = +pos[0] * np.cos(-self.angle) + pos[1] * np.sin(-self.angle)
            y = -pos[0] * np.sin(-self.angle) + pos[1] * np.cos(-self.angle)
            u = self.source_to_detector / (x + self.source_to_image) * y
            return self.detector_attr.locate(u)
        elif isinstance(self.detector_attr, Detector2DAttr):
            if len(pos) > 3:
                raise ValueError
            x = +pos[0] * np.cos(-self.angle) + pos[1] * np.sin(-self.angle)
            y = -pos[0] * np.sin(-self.angle) + pos[1] * np.cos(-self.angle)
            z = pos[2]
            u = self.source_to_detector / (x + self.source_to_image) * y
            v = self.source_to_detector / (x + self.source_to_image) * z
            return self.detector_attr.locate((u, v))
        else:
            raise NotImplementedError


class ProjectionCurveAttr(ProjectionAttr):
    def detector_unit_centers(self):
        if isinstance(self.detector_attr, Detector1DAttr):
            u = self.detector_attr.unit_centers()
            xd, yd = np.cos(u) * self.source_to_detector - self.source_to_image, \
                     np.sin(u) * self.source_to_detector
            x = +np.cos(self.angle) * xd - np.sin(self.angle) * yd
            y = -np.sin(self.angle) * xd + np.cos(self.angle) * yd
            return x, y
        elif isinstance(self.detector_attr, Detector2DAttr):
            u, v = self.detector_attr.unit_centers()
            xd, yd = np.cos(u) * self.source_to_detector - self.source_to_image, \
                     np.sin(u) * self.source_to_detector
            x = +np.cos(self.angle) * xd - np.sin(self.angle) * yd
            y = -np.sin(self.angle) * xd + np.cos(self.angle) * yd
            z = v
            return x, y, z
        else:
            raise NotImplementedError

    def locate(self, pos):
        if isinstance(self.detector_attr, Detector1DAttr):
            if len(pos) > 2:
                raise ValueError
            x = +pos[0] * np.cos(-self.angle) + pos[1] * np.sin(-self.angle)
            y = -pos[0] * np.sin(-self.angle) + pos[1] * np.cos(-self.angle)
            u = np.arctan2(y, x + self.source_to_image)
            return self.detector_attr.locate(u)
        elif isinstance(self.detector_attr, Detector2DAttr):
            if len(pos) > 3:
                raise ValueError
            x = +pos[0] * np.cos(-self.angle) + pos[1] * np.sin(-self.angle)
            y = -pos[0] * np.sin(-self.angle) + pos[1] * np.cos(-self.angle)
            z = pos[2]
            u = np.arctan2(y, x + self.source_to_image)
            v = self.source_to_image / (x + self.source_to_detector) * z
            return self.detector_attr.locate((u, v))
        else:
            raise None
