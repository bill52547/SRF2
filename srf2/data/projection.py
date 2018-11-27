import h5py
import numpy as np
from collections import List
from ..core.abstracts import Object
from ..attrs.projectionattr import *

__all__ = ('Projection', 'ProjectionSeries', 'ProjectionFlatAttr', 'ProjectionCurveAttr',)


class Projection(Object):
    _data: np.ndarray
    _attr: ProjectionAttr

    def __init__(self, data=None, attr: ProjectionAttr = None):
        super().__init__(data, attr)

    @property
    def data(self):
        return self._data

    @property
    def attr(self):
        return self._attr

    def map(self, f):
        return self.__class__(*f(self.data, self.attr))

    def __getitem__(self, item):
        def _getitem(item):
            def kernel(data, attr):
                return data[item], attr[item]

            return kernel

        return self.map(_getitem(item))


class ProjectionSeries:
    _data: List[np.ndarray]
    _attr: List[ProjectionAttr]

    def __init__(self, data: List[np.ndarray], attr: List[ProjectionAttr]):
        self._data = data
        self._attr = attr

    @property
    def data(self):
        return self._data

    @property
    def attr(self):
        return self._attr

    def map(self, f):
        return self.__class__(*f(self.data, self.attr))

    def __getitem__(self, item):
        def _getitem(item):
            item1, item2 = item[:-1], item[-1]

            def kernel(data, attr):
                return [d[item1] for d in data[item2]], [a[item1] for a in attr[item2]]

            return kernel

        return self.map(_getitem(item))

    def __len__(self):
        return len(self.attr)


class ProjectionFlatSeries(ProjectionSeries):
    _data: List[np.ndarray]
    _attr: List[ProjectionFlatAttr]

    def __init__(self, data: List[np.ndarray], attr: List[ProjectionFlatAttr]):
        super().__init__(data, attr)
        if attr[0] is not ProjectionFlatAttr:
            raise ValueError


class ProjectionCurveSeries(ProjectionSeries):
    _data: List[np.ndarray]
    _attr: List[ProjectionCurveAttr]

    def __init__(self, data: List[np.ndarray], attr: List[ProjectionCurveAttr]):
        super().__init__(data, attr)
        if attr[0] is not ProjectionCurveAttr:
            raise ValueError


