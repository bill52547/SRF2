from collections import List

import numpy as np

from ..attr.projection_attr import *
from ..core.abstracts import Object

__all__ = ('Projection', 'ProjectionFlat', 'ProjectionCurve',
           'ProjectionSeries', 'ProjectionFlatSeries', 'ProjectionCurveSeries',)


class Projection(Object):
    def __init__(self, attr: ProjectionAttr, data: np.ndarray = None):
        if attr is None:
            raise ValueError
        self._attr = attr
        self._data = data
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
        return self.__class__(self.attr[item], self.data[item])

    def squeeze(self):
        attr = self.attr.squeeze()
        data = self.data.squeeze()
        if isinstance(attr, ProjectionCurveAttr):
            return ProjectionCurve(attr, data)
        elif isinstance(attr, ProjectionFlatAttr):
            return ProjectionFlat(attr, data)
        else:
            return Projection(attr, data)


class ProjectionCurve(Projection):
    def __init__(self, attr: ProjectionCurveAttr, data: np.ndarray = None):
        super().__init__(attr, data)


class ProjectionFlat(Projection):
    def __init__(self, attr: ProjectionFlatAttr, data: np.ndarray = None):
        super().__init__(attr, data)


class ProjectionSeries(Object):
    def __init__(self, attr: List[ProjectionAttr], data: List[np.ndarray]):
        self._attr = [a for a in attr]
        self._data = [d for d in data]

    def __len__(self):
        return len(self.data)

    def map(self, f):
        return self.__class__(*f(self.attr, self.data))

    def __getitem__(self, item):
        return self.__class__(self.attr[item], self.data[item])

    def squeeze(self):
        if self.__len__() == 1:
            if isinstance(self.attr[0], ProjectionCurveAttr):
                return ProjectionCurve(self.attr[0], self.data[0])
            elif isinstance(self.attr[0], ProjectionFlatAttr):
                return ProjectionFlat(self.attr[0], self.data[0])
            else:
                raise NotImplementedError
        attr = [a.squeeze() for a in self.attr]
        data = [d.squeeze() for d in self.data]
        return self.__class__(attr, data)


class ProjectionFlatSeries(ProjectionSeries):
    def __init__(self, attr: List[ProjectionFlatAttr], data: List[np.ndarray]):
        super().__init__(attr, data)


class ProjectionCurveSeries(ProjectionSeries):
    def __init__(self, attr: List[ProjectionCurveAttr], data: List[np.ndarray]):
        super().__init__(attr, data)
