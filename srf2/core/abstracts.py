#!/usr/bin/env python
# encoding: utf-8
'''
@author: Minghao Guo, Xiang Hong, Gaoyu Chen and Weijie Tao
@license: LGPL_v3.0
@contact: mh.guo0111@gmail.com
@software: srf_v2
@file: abstract.py
@date: 11/10/2018
@desc: new version of Scalable Reconstruction Framework for Medical Imaging
'''

from abc import abstractmethod

import h5py
import numpy as np
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray as DeviceNDArray
from numpy.core import isscalar

from .cuda_arithmetics import *
from .type_assert import *

__all__ = ('Attribute', 'AttributeWithShape', 'ObjectWithAttrData',)


def _encode_utf8(val):
    if isinstance(val, tuple):
        return tuple([v.encode('utf-8') if isinstance(v, str) else v for v in val])
    else:
        return val.encode('utf-8') if isinstance(val, str) else val


def _decode_utf8(val):
    if isinstance(val, (list, np.ndarray)):
        return tuple([v.decode('utf-8') if isinstance(v, bytes) else v for v in val])
    else:
        return val.decode('utf-8') if isinstance(val, bytes) else val


class Attribute(object):
    ''' An base attibute class.
    A attribute is an object who consist the attribute to describe another object. More specific, a
    attribute object only contains small descriptions, which can only be a tuple or a value/string (
    <64k) and can be stored in a hdf5 attibute.
    '''
    __device_manager__ = 'host'

    def __eq__(self, other):
        '''**Equality verify**
        :param other:
        :return: bool
        '''
        if not isinstance(other, self.__class__):
            return False
        for key, value in self.__dict__.items():
            if value != other.__getattribute__(key):
                return False
        else:
            return True

    def copy(self):
        dict_attrs = {}
        for key, value in self.__dict__.items():
            if key.startswith('__'):
                continue
            if key.startswith('_'):
                key1 = key[1:]
            else:
                key1 = key
            dict_attrs[key1] = _decode_utf8(value)
        attr = self.__class__(**dict_attrs)
        attr.__device_manager__ = self.__device_manager__
        return attr

    def save_h5(self, path=None, mode='w'):
        '''**save to hdf5 file**
        save a attribute object to hdf5 file, in term of hdf5 group/attrs. It is saved in a group
        with name of this class.
        :param path: should be end with '.h5' or 'hdf5'
        :param mode: 'w' by default.
        :return: None
        '''

        if path is None:
            path = 'tmp' + self.__class__.__name__ + '.h5'

        if not str.endswith(path, 'h5') and not str.endswith(path, 'hdf5'):
            raise ValueError(self.__class__.save_h5.__qualname__,
                             ' should have path input ends with h5 or hdf5')
        with h5py.File(path, mode) as fout:
            group = fout.create_group(self.__class__.__name__)
            for key, value in self.__dict__.items():
                if key.startswith('__'):
                    continue
                if key.startswith('_'):
                    key1 = key[1:]
                else:
                    key1 = key
                group.attrs.create(key1, data=_encode_utf8(value))

    @classmethod
    def load_h5(cls, path=None):
        if cls is Attribute:
            return NotImplementedError

        if path is None:
            path = 'tmp' + cls.__name__ + '.h5'
        with h5py.File(path, 'r') as fin:
            group = fin[cls.__name__]
            dict_attrs = {}
            for key, value in group.attrs.items():
                dict_attrs[key] = _decode_utf8(value)
            return cls(**dict_attrs)

    def __repr__(self):
        out_str = f'{type(self)} object at {hex(id(self))}\n'
        for key in self.__dict__.keys():
            out_str += f'{key}: {type(self.__dict__[key])} = {self.__dict__[key]}\n'
        return out_str

    __str__ = __repr__

    @abstractmethod
    def map(self, _):
        raise NotImplementedError('map method is valid for ', self.__class__, ' object.')


class AttributeWithShape(Attribute):
    _shape = None

    @arg_type_assert(None, tuple)
    def __init__(self, shape=None):
        if shape is None:
            raise ValueError
        self._shape = shape

    @property
    def shape(self):
        return self._shape


class ObjectWithAttrData(object):
    _attr = None
    __device_manager__ = 'host'

    @arg_type_assert(None, AttributeWithShape)
    def __init__(self, attr, data=None):
        self._attr = attr
        self._data = data if data is not None else np.zeros(attr.shape)
        if isinstance(self._data, np.ndarray):
            self.__device_manager__ = 'host'
        elif isinstance(self._data, DeviceNDArray):
            self.__device_manager__ = 'gpu'

    @property
    def attr(self):
        return self._attr

    @property
    def data(self):
        return self._data

    def copy(self, stream=0):
        if self.__device_manager__ == 'gpu':
            h_data = self.data.copy_to_host(stream=stream)
            d_data = cuda.to_device(h_data, stream=stream)
            return self.__class__(self.attr, d_data)
        elif self.__device_manager__ == 'host':
            h_data = self.data.copy()
            return self.__class__(self.attr, h_data)
        else:
            raise NotImplementedError

    def to_device(self, stream=0):
        if self.__device_manager__ == 'gpu':
            return self
        elif self.__device_manager__ == 'host':
            self._data = cuda.to_device(self.data, stream)
            self.__device_manager__ = 'gpu'
            return self
        else:
            raise NotImplementedError

    def to_host(self, stream=0):
        if self.__device_manager__ == 'host':
            return self
        elif self.__device_manager__ == 'gpu':
            if stream is None:
                self._data = self.data.copy_to_host(stream=stream)
            self.__device_manager__ = 'host'
            return self
        else:
            raise NotImplementedError

    def to_target(self, target=None, stream=0):
        if target is None:
            return self
        elif target == 'gpu':
            self.to_device(stream)
            return self
        elif target == 'host':
            self.to_host(stream)
            return self
        else:
            raise NotImplementedError

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.attr != other.attr:
            return False

        if not self and not other:
            return True
        elif not self or not other:
            return False
        if self.__device_manager__ == 'host':
            return np.array_equal(other.data, self.data)
        else:
            raise NotImplementedError

    def save_h5(self, path=None, mode='w'):
        '''**save to hdf5 file**
        save a attribute object to hdf5 file, in term of hdf5 group/attrs. It is saved in a group
        with name of this class.
        :param path: should be end with '.h5' or 'hdf5'
        :param mode: 'w' by default.
        :return: None
        '''

        if path is None:
            path = 'tmp' + self.__class__.__name__ + '.h5'
        if not str.endswith(path, 'h5') and not str.endswith(path, 'hdf5'):
            raise ValueError(self.__class__.save_h5.__qualname__,
                             ' should have path input ends with h5 or hdf5')
        self.to_host()
        self.attr.save_h5(path, mode)
        with h5py.File(path, 'r+') as fout:
            fout.create_dataset('data', data=self.data, compression="gzip")

    @classmethod
    def load_h5(cls, path=None):
        if cls is ObjectWithAttrData:
            return NotImplementedError

        attr = cls._attr.__class__.load_h5(path)
        with h5py.File(path, 'r') as fin:
            data = np.array(fin['data'])
            return cls(attr, data)

    def map(self, f):
        return self.__class__(*f(self.attr, self.data))

    def __repr__(self):
        out_str = f'{type(self)} object at {hex(id(self))} with attributes as:\n'
        out_str += self.attr.__repr__()
        return out_str

    def __neg__(self, stream=0):
        if self.__device_manager__ == 'host':
            return self.__class__(self.attr, -self.data)
        elif self.__device_manager__ == 'gpu':
            return self.__mul__(-1, stream)
        else:
            raise NotImplementedError

    def __pos__(self, stream=0):
        return self.copy(stream)

    def __add__(self, other, stream=0):
        obj = self.copy(stream)
        obj.__iadd__(other, stream)
        return obj

    def __radd__(self, other, stream=0):
        if isinstance(other, ObjectWithAttrData):
            if self.__device_manager__ == other.__device_manager__:
                return self.__add__(other, stream)
            else:
                raise NotImplementedError
        else:
            return self.__add__(other, stream)

    def __sub__(self, other, stream=0):
        obj = self.copy(stream)
        obj.__isub__(other, stream)
        return obj

    def __mul__(self, other, stream=0):
        obj = self.copy(stream)
        obj.__imul__(other, stream)
        return obj

    def __rmul__(self, other, stream=0):
        if isinstance(other, ObjectWithAttrData):
            if self.__device_manager__ == other.__device_manager__:
                return self.__mul__(other, stream)
            else:
                raise NotImplementedError
        else:
            return self.__mul__(other, stream)

    def __truediv__(self, other, stream=0):
        obj = self.copy(stream)
        obj.__itruediv__(other, stream)
        return obj

    def __iadd__(self, other, stream=0):
        if self.__device_manager__ == 'host':
            if isscalar(other) or isinstance(other, np.ndarray):
                self._data += other
            elif isinstance(other, self.__class__):
                self._data += other.to_host().data
            else:
                raise NotImplementedError
        elif self.__device_manager__ == 'gpu':
            if isinstance(other, self.__class__):
                cuda_iadd_with_array(self._data, other.to_device(stream).data, stream)
            elif isinstance(other, DeviceNDArray):
                cuda_iadd_with_array(self._data, other, stream)
            elif isinstance(other, np.ndarray):
                cuda_iadd_with_array(self._data, cuda.to_device(other, stream), stream)
            elif isscalar(other):
                cuda_iadd_with_scale(self._data, other, stream)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return self

    def __isub__(self, other, stream=0):
        if self.__device_manager__ == 'host':
            if isscalar(other) or isinstance(other, np.ndarray):
                self._data -= other
            elif isinstance(other, self.__class__):
                self._data -= other.to_host().data
            else:
                raise NotImplementedError
        elif self.__device_manager__ == 'gpu':
            if isinstance(other, self.__class__):
                cuda_isub_with_array(self._data, other.to_device(stream).data, stream)
            elif isinstance(other, DeviceNDArray):
                cuda_isub_with_array(self._data, other, stream)
            elif isinstance(other, np.ndarray):
                cuda_isub_with_array(self._data, cuda.to_device(other, stream), stream)
            elif isscalar(other):
                cuda_isub_with_scale(self._data, other, stream)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return self

    def __imul__(self, other, stream=0):
        if self.__device_manager__ == 'host':
            if isscalar(other) or isinstance(other, np.ndarray):
                self._data *= other
            elif isinstance(other, self.__class__):
                self._data *= other.to_host().data
            else:
                raise NotImplementedError
        elif self.__device_manager__ == 'gpu':
            if isinstance(other, self.__class__):
                cuda_imul_with_array(self._data, other.to_device(stream).data, stream)
            elif isinstance(other, DeviceNDArray):
                cuda_imul_with_array(self._data, other, stream)
            elif isinstance(other, np.ndarray):
                cuda_imul_with_array(self._data, cuda.to_device(other, stream), stream)
            elif isscalar(other):
                cuda_imul_with_scale(self._data, other, stream)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return self

    def __itruediv__(self, other, stream=0):
        if self.__device_manager__ == 'host':
            if isscalar(other) or isinstance(other, np.ndarray):
                self._data /= other
            elif isinstance(other, self.__class__):
                self._data /= other.to_host().data
            else:
                raise NotImplementedError
        elif self.__device_manager__ == 'gpu':
            if isinstance(other, self.__class__):
                cuda_itruediv_with_array(self._data, other.to_device(stream).data, stream)
            elif isinstance(other, DeviceNDArray):
                cuda_itruediv_with_array(self._data, other, stream)
            elif isinstance(other, np.ndarray):
                cuda_itruediv_with_array(self._data, cuda.to_device(other), stream)
            elif isscalar(other):
                cuda_itruediv_with_scale(self._data, other, stream)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return self
