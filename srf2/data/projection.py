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
    def __init__(self, attr: ProjectionSeriesAttr, data: np.ndarray = None):
        self._attr = attr
        self._data = data
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise ValueError
            self._data.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __bool__(self):
        return not self.data and len(self) > 0

    def map(self, f):
        return self.__class__(*f(self.attr, self.data))

    def __getitem__(self, item):
        if len(item) == 1:
            item1, item2 = [None] * (self.data.ndim - 1), item
        elif len(item) == self.data.ndim:
            item1, item2 = item[:-1], item[-1]
        else:
            raise NotImplementedError
        if self:
            return self.__class__(self.attr[item2], self.data[[item1, item2]])
        else:
            return self.__class__(self.attr[item2], None)

    def squeeze(self):
        return self.__class__(self.attr.squeeze(), self.data.squeeze())


class ProjectionFlatSeries(ProjectionSeries):
    def __init__(self, attr: ProjectionFlatSeriesAttr, data: np.ndarray = None):
        super().__init__(attr, data)


class ProjectionCurveSeries(ProjectionSeries):
    def __init__(self, attr: ProjectionCurveSeriesAttr, data: np.ndarray = None):
        super().__init__(attr, data)
