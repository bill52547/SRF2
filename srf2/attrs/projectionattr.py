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
from abc import abstractmethod

from srf2.core.abstracts import Attribute

__all__ = ('Detector1DAttr', 'Detector2DAttr', 'ProjectionFlatAttr', 'ProjectionCurveAttr',)


class DetectorAttr(Attribute):
    _shape: tuple
    _center: tuple
    _size: tuple

    def __init__(self, shape, center, size):
        if not (len(shape) == len(center) == len(size)):
            raise ValueError(self.__dict__, ' should have same lengths')
        if len(shape) < 1 or len(shape) > 2:
            raise NotImplemented
        self._shape = shape
        self._center = center
        self._size = size

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
    def n_u(self):
        return self.shape[0]

    @property
    def n_v(self):
        return self.shape[1] if len(self.shape) > 1 else 1

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def num_all(self):
        return self.n_u * self.n_v

    @abstractmethod
    def meshgrid(self):
        pass

    @abstractmethod
    def unit_centers(self):
        pass

    def map(self, f):
        return self.__class__(*f(self.shape, self.center, self.size))

    def __getitem__(self, item):
        shape = list(self.shape)
        center = list(self.center)
        size = list(self.size)

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

        count_nonzero = shape.count(1)
        if count_nonzero == 1:
            return Detector1DAttr(shape, center, size)
        elif count_nonzero == 2:
            return Detector2DAttr(shape, center, size)
        else:
            raise NotImplementedError

    def locate(self, pos=None):
        if pos is None:
            return ValueError('No valid input.')
        result = [0] * self.ndim
        for k in range(self.ndim):
            result[k] = (pos[k] - self.center[k] + self.size[k] / 2) / self.unit_size[k] - 0.5
        return tuple(result)


class Detector1DAttr(DetectorAttr):
    def __init__(self, shape=(1,), center=(0,), size=(1,)):
        super().__init__(shape, center, size)
        if len(shape) != 1:
            raise ValueError(self.__class__, ' is only consistent with 2D case')

    def meshgrid(self):
        return np.arange(self.shape[0])

    def unit_centers(self):
        return self.meshgrid() * self.unit_size[0] + self.center[0] - self.size[0] / 2 + self.unit_size[0] / 2


class Detector2DAttr(DetectorAttr):
    def __init__(self, shape=(1, 1), center=(0, 0), size=(1, 1)):
        super().__init__(shape, center, size)
        if len(shape) != 2:
            raise ValueError(self.__class__, ' is only consistent with 2D case')

    def map(self, f):
        return Detector2DAttr(*f(self.shape, self.center, self.size))

    def meshgrid(self):
        x = np.arange(self.shape[0])
        y = np.arange(self.shape[1])
        y1, x1 = np.meshgrid(y, x)
        return x1, y1

    def unit_centers(self):
        x1, y1 = self.meshgrid()
        pos_x = x1 * self.unit_size[0] + self.center[0] - self.size[0] / 2 + self.unit_size[0] / 2
        pos_y = y1 * self.unit_size[1] + self.center[1] - self.size[1] / 2 + self.unit_size[1] / 2
        return pos_x, pos_y


class ProjectionAttr(Attribute):
    _SID: np.float32
    _SAD: np.float32
    _angle: np.float32
    _detector_attr: DetectorAttr
    _pos_z: np.float32

    def __init__(self, SID, SAD, angle, detector_attr: DetectorAttr, pos_z=0):
        self._SID = np.float32(SID)
        self._SAD = np.float32(SAD)
        self._angle = np.float32(angle)
        self._detector_attr = detector_attr
        self._pos_z = np.float32(pos_z)

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
    def detector_attr(self):
        return self._detector_attr

    @property
    def pos_z(self):
        return self._pos_z

    @property
    def source_pos(self):
        x, y = -self.SID * np.cos(self.angle), -self.SID * np.sin(self.angle)
        if self.detector_attr is Detector1DAttr:
            return x, y
        else:
            return x, y, self.pos_z

    @abstractmethod
    def unit_centers(self):
        pass

    @abstractmethod
    def locate(self, pos):
        pass

    def map(self, f):
        return self.__class__(*f(self.SID, self.SAD, self.detector_attr, self.detector_attr, self.pos_z))


class ProjectionFlatAttr(ProjectionAttr):
    def unit_centers(self):
        if self.detector_attr is Detector1DAttr:
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
        if self.detector_attr is Detector1DAttr:
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


class ProjectionCurveAttr(ProjectionAttr):
    def unit_centers(self):
        if self.detector_attr is Detector1DAttr:
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
        if self.detector_attr is Detector1DAttr:
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
