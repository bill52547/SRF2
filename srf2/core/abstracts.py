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

__all__ = ('Attribute', 'ObjectWithAttrData',)



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
    memory_type = 'cpu'

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

    def to_device(self):
        self.memory_type = 'gpu'

    def to_host(self):
        self.memory_type = 'cpu'

    def copy(self):
        dict_attrs = {}
        for key, value in self.__dict__.items():
            if key.startswith('_'):
                key1 = key[1:]
            else:
                key1 = key
            dict_attrs[key1] = _decode_utf8(value)
        return self.__class__(**dict_attrs)

    def save_h5(self, path = None, mode = 'w'):
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
                if key.startswith('_'):
                    key1 = key[1:]
                else:
                    key1 = key
                group.attrs.create(key1, data = _encode_utf8(value))

    @classmethod
    def load_h5(cls, path = None):
        if cls is Attribute:
            return NotImplementedError

        if path is None:
            path = 'tmp' + cls.__name__ + '.h5'
        with h5py.File(path, 'r') as fin:
            group = fin[cls.__name__]
            dict_attrs = {}
            for key, value in group.attrs.items():
                if key.startswith('_'):
                    key1 = key[1:]
                else:
                    key1 = key
                dict_attrs[key1] = _decode_utf8(value)
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


class ObjectWithAttrData(object):

    _attr = None

    @arg_type_assert(None, Attribute)
    def __init__(self, attr, data = None):
        self._attr = attr
        self._data = data

    @property
    def attr(self):
        return self._attr

    @property
    def data(self):
        return self._data

    def to_device(self, stream = None):
        if self.attr.memory_type == 'gpu':
            return
        if stream is None:
            stream = cuda.stream()
        self._attr.to_device()
        self._data = cuda.to_device(self.data, stream)

    def to_host(self, stream = None):
        if self.attr.memory_type == 'cpu':
            return
        if stream is None:
            stream = cuda.stream()
        self._attr.to_host()
        self._data = self.data.copy_to_host(stream = stream)

    def to_target(self, target = None, stream = None):
        if target is None:
            return
        if isinstance(target, self.__class__):
            if target.attr.memory_type == 'cpu':
                self.to_host(stream)
            elif target.attr.memory_type == 'gpu':
                self.to_device(stream)
            else:
                raise NotImplementedError
        elif target == 'cpu':
            self.to_host(stream)
        elif target == 'gpu':
            self.to_device(stream)
        else:
            raise NotImplementedError

    def copy(self):
        if self.attr.memory_type == 'gpu':
            h_data = self.data.copy_to_host().copy()
            d_data = cuda.to_device(h_data)
            return self.__class__(self.attr.copy(), d_data)
        elif self.attr.memory_type == 'cpu':
            h_data = self.data.copy()
            return self.__class__(self.attr.copy(), h_data)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.attr != other.attr:
            return False

        if not self and not other:
            return True
        elif not self or not other:
            return False
        if self.attr.memory_type == 'cpu':
            return np.array_equal(other.data, self.data)
        else:
            raise NotImplementedError

    def save_h5(self, path = None, mode = 'w'):
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
            fout.create_dataset('data', data = self.data, compression = "gzip")

    @classmethod
    def load_h5(cls, path = None):
        if cls is ObjectWithAttrData:
            return NotImplementedError

        attr = cls._attr.__class__.load_h5(path)
        with h5py.File(path, 'r') as fin:
            data = np.array(fin['data'])
            return cls(attr, data)

    def map(self, f):
        return self.__class__(*f(self.attr, self.data))

    def __repr__(self):
        out_str = f'{type(self)} object at {hex(id(self))}:\n'
        out_str += 'with attributes as:\n'
        out_str += self.attr.__repr__()
        return out_str

    def __neg__(self):
        if self.attr.memory_type == 'cpu':
            def _neg(attr, data):
                return attr, -data

            return self.copy().map(_neg)
        elif self.attr.memory_type == 'gpu':
            return self.copy().__imul__(-1)

    def __pos__(self):
        if self.attr.memory_type == 'cpu':
            return self.copy()
        elif self.attr.memory_type == 'gpu':
            return self.copy()

    def __add__(self, other):
        obj = self.copy()
        obj += other
        return obj

    def __radd__(self, other):
        if self.attr.memory_type == 'cpu':
            return self + other
        else:
            raise NotImplementedError

    def __sub__(self, other):
        obj = self.copy()
        obj -= other
        return obj

    def __mul__(self, other):
        obj = self.copy()
        obj *= other
        return obj

    def __rmul__(self, other):
        if self.attr.memory_type == 'cpu':
            return self * other
        else:
            raise NotImplementedError

    def __truediv__(self, other):
        obj = self.copy()
        obj /= other
        return obj

    def __iadd__(self, other):
        if self.attr.memory_type == 'cpu':
            if isscalar(other) or isinstance(other, np.ndarray):
                self._data += other
            elif isinstance(other, self.__class__):
                other.to_host()
                self._data += other.data
            elif isinstance(other, DeviceNDArray):
                self._data += other.copy_to_host()
            else:
                raise NotImplementedError
        elif self.attr.memory_type == 'gpu':
            if isinstance(other, self.__class__):
                if other.attr.memory_type == 'cpu':
                    data1 = cuda.to_device(other.data)
                else:
                    data1 = other._data
                cuda_iadd_with_array(self._data, data1)
            elif isinstance(other, DeviceNDArray):
                cuda_iadd_with_array(self._data, other)
            elif isinstance(other, np.ndarray):
                data1 = cuda.to_device(other)
                cuda_iadd_with_array(self._data, data1)
            elif isscalar(other):
                cuda_iadd_with_scale(self._data, other)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return self

    def __isub__(self, other):
        if self.attr.memory_type == 'cpu':
            if isscalar(other) or isinstance(other, np.ndarray):
                self._data -= other
            elif isinstance(other, self.__class__):
                other.to_host()
                self._data -= other.data
            elif isinstance(other, DeviceNDArray):
                self._data -= other.copy_to_host()
            else:
                raise NotImplementedError
        elif self.attr.memory_type == 'gpu':
            if isinstance(other, self.__class__):
                if other.attr.memory_type == 'cpu':
                    data1 = cuda.to_device(other.data)
                else:
                    data1 = other._data
                cuda_isub_with_array(self._data, data1)
            elif isinstance(other, DeviceNDArray):
                cuda_isub_with_array(self._data, other)
            elif isinstance(other, np.ndarray):
                data1 = cuda.to_device(other)
                cuda_isub_with_array(self._data, data1)
            elif isscalar(other):
                cuda_isub_with_scale(self._data, other)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return self

    def __imul__(self, other):
        if self.attr.memory_type == 'cpu':
            if isscalar(other) or isinstance(other, np.ndarray):
                self._data *= other
            elif isinstance(other, self.__class__):
                other.to_host()
                self._data *= other.data
            elif isinstance(other, DeviceNDArray):
                self._data *= other.copy_to_host()
            else:
                raise NotImplementedError
        elif self.attr.memory_type == 'gpu':
            if isinstance(other, self.__class__):
                if other.attr.memory_type == 'cpu':
                    data1 = cuda.to_device(other.data)
                else:
                    data1 = other._data
                cuda_imul_with_array(self._data, data1)
            elif isinstance(other, DeviceNDArray):
                cuda_imul_with_array(self._data, other)
            elif isinstance(other, np.ndarray):
                data1 = cuda.to_device(other)
                cuda_imul_with_array(self._data, data1)
            elif isscalar(other):
                cuda_imul_with_scale(self._data, other)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return self

    def __itruediv__(self, other):
        if self.attr.memory_type == 'cpu':
            if isscalar(other) or isinstance(other, np.ndarray):
                self._data /= other
            elif isinstance(other, self.__class__):
                other.to_host()
                self._data /= other.data
            elif isinstance(other, DeviceNDArray):
                self._data /= other.copy_to_host()
            else:
                raise NotImplementedError
        elif self.attr.memory_type == 'gpu':
            if isinstance(other, self.__class__):
                if other.attr.memory_type == 'cpu':
                    data1 = cuda.to_device(other.data)
                else:
                    data1 = other._data
                cuda_itruediv_with_array(self._data, data1)
            elif isinstance(other, DeviceNDArray):
                cuda_itruediv_with_array(self._data, other)
            elif isinstance(other, np.ndarray):
                data1 = cuda.to_device(other)
                cuda_itruediv_with_array(self._data, data1)
            elif isscalar(other):
                cuda_itruediv_with_scale(self._data, other)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return self
