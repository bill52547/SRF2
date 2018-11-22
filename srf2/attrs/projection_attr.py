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

import numpy as np
from functools import reduce
from abc import abstractmethod

from srf2.core.abstracts import Attribute

__all__ = ('Detector_1d_attr', 'Detector_2d_attr', 'Projection_flat_attr', 'Projection_curve_attr',)


class Detector_attr(Attribute):
    def __init__(self, shape, center, size):
        self._shape = shape
        self._center = center
        self._size = size
        if not (len(self._shape) == len(self._center) == len(self._size)):
            raise ValueError(self.__dict__, ' should have same lengths')
        if len(self._shape) < 1 or len(self._shape) > 2:
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
    def unit_size(self):
        return tuple([x / y for (x, y) in zip(self.size, self.shape)])

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def num_all(self):
        return reduce(lambda x, y: x * y, self.shape)

    @abstractmethod
    def meshgrid(self):
        pass

    @abstractmethod
    def unit_centers(self):
        pass

    @abstractmethod
    def map(self, f):
        pass

    def __getitem__(self, ind):
        shape = list(self.shape)
        center = list(self.center)
        size = list(self.size)

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
            return Detector_1d_attr(shape, center, size)
        elif count_nonzero == 2:
            return Detector_2d_attr(shape, center, size)
        else:
            raise NotImplementedError

    def locate(self, pos=None):
        if pos is None:
            return ValueError('No valid input.')
        result = [0] * self.ndim
        for k in range(self.ndim):
            result[k] = (pos[k] - self.center[k] + self.size[k] / 2) / self.unit_size[k] - 0.5
        return tuple(result)

    # def transpose(self, perm=None):
    #     if perm is None:
    #         perm = np.arange(self.ndim)[::-1]
    #     if set(perm).issubset({'a', 'b'}):
    #         perm = [self.dims.index(e) for e in perm]
    #     shape = [self.shape[i] for i in perm]
    #     center = [self.center[i] for i in perm]
    #     size = [self.size[i] for i in perm]
    #     dims = [self.dims[i] for i in perm]
    #     return self.__class__(shape, center, size, dims)


class Detector_1d_attr(Detector_attr):
    def __init__(self, shape=(1,), center=(0,), size=(1,)):
        super().__init__(shape, center, size)
        if len(shape) != 1:
            raise ValueError(self.__class__, ' is only consistent with 2D case')

    def map(self, f):
        return Detector_1d_attr(*f(self.shape, self.center, self.size))

    def meshgrid(self):
        return np.arange(self.shape[0])

    def unit_centers(self):
        return self.meshgrid() * self.unit_size[0] + self.center[0] - self.size[0] / 2 + self.unit_size[0] / 2


class Detector_2d_attr(Detector_attr):
    def __init__(self, shape=(1, 1), center=(0, 0), size=(1, 1)):
        super().__init__(shape, center, size)
        if len(shape) != 2:
            raise ValueError(self.__class__, ' is only consistent with 2D case')

    def map(self, f):
        return Detector_2d_attr(*f(self.shape, self.center, self.size))

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


class Projection_attr(Attribute):
    def __init__(self, SID, SAD, angle, detector_attr: Detector_attr, pos_z=0):
        self._SID = SID
        self._SAD = SAD
        self._angle = angle
        self.detector_attr = detector_attr
        self._pos_z = pos_z

    @property
    def SID(self):
        return self._SID

    @property
    def SAD(self):
        return self._SAD

    @property
    def angle(self):
        return self._angle

    @property
    def pos_z(self):
        return self._pos_z

    @property
    def source_pos(self):
        x, y = -self.SID * np.cos(self.angle), -self.SID * np.sin(self.angle)
        if self.detector_attr is Detector_1d_attr:
            return x, y
        else:
            return x, y, self.pos_z

    @abstractmethod
    def unit_centers(self):
        pass

    @abstractmethod
    def locate(self, pos):
        pass

    @abstractmethod
    def map(self, f):
        pass


class Projection_flat_attr(Projection_attr):
    def unit_centers(self):
        if self.detector_attr is Detector_1d_attr:
            a = self.detector_attr.unit_centers()
            x = np.cos(self.angle) * (self.SID - self.SAD) - np.sin(self.angle) * a
            y = np.sin(self.angle) * a
            return x, y
        else:
            a, b = self.detector_attr.unit_centers()
            x = np.cos(self.angle) * (self.SID - self.SAD) - np.sin(self.angle) * a
            y = np.sin(self.angle) * a
            z = b + self.pos_z
            return x, y, z

    def locate(self, pos):
        if self.detector_attr is Detector_1d_attr:
            if len(pos) > 2 and pos[2] != self.pos_z:
                raise ValueError
            x = +pos[0] * np.cos(-self.angle) + pos[1] * np.sin(-self.angle)
            y = -pos[0] * np.sin(-self.angle) + pos[1] * np.cos(-self.angle)
            a = self.SAD / (x + self.SID) * y
            return self.detector_attr.locate(a)
        else:
            x = +pos[0] * np.cos(-self.angle) + pos[1] * np.sin(-self.angle)
            y = -pos[0] * np.sin(-self.angle) + pos[1] * np.cos(-self.angle)
            z = pos[2]
            a = self.SAD / (x + self.SID) * y
            b = self.SAD / (x + self.SID) * z + self.pos_z
            return self.detector_attr.locate((a, b))

    def map(self, f):
        pass


class Projection_curve_attr(Projection_attr):
    def unit_centers(self):
        if self.detector_attr is Detector_1d_attr:
            a = self.detector_attr.unit_centers()
            xd, yd = np.cos(a) * self.SID - self.SAD, np.sin(a) * self.SID
            x = +np.cos(self.angle) * xd - np.sin(self.angle) * yd
            y = -np.sin(self.angle) * xd + np.cos(self.angle) * yd
            return x, y
        else:
            a, b = self.detector_attr.unit_centers()
            xd, yd = np.cos(a) * self.SID - self.SAD, np.sin(a) * self.SID
            x = +np.cos(self.angle) * xd - np.sin(self.angle) * yd
            y = -np.sin(self.angle) * xd + np.cos(self.angle) * yd
            z = b + self.pos_z
            return x, y, z

    def locate(self, pos):
        if self.detector_attr is Detector_1d_attr:
            if len(pos) > 2 and pos[2] != self.pos_z:
                raise ValueError
            x = +pos[0] * np.cos(-self.angle) + pos[1] * np.sin(-self.angle)
            y = -pos[0] * np.sin(-self.angle) + pos[1] * np.cos(-self.angle)
            a = np.arctan2(y, x + self.SAD)
            return self.detector_attr.locate(a)
        else:
            x = +pos[0] * np.cos(-self.angle) + pos[1] * np.sin(-self.angle)
            y = -pos[0] * np.sin(-self.angle) + pos[1] * np.cos(-self.angle)
            z = pos[2]
            a = np.arctan2(y, x + self.SAD)
            b = self.SAD / (x + self.SID) * z + self.pos_z
            return self.detector_attr.locate((a, b))

    def map(self, f):
        pass
