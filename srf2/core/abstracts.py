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
import numpy as np
from numpy.core import isscalar
import h5py

__all__ = ('Attribute', 'Object')


def _encode_utf8(val):
    if val is tuple:
        return tuple([v.encode('utf-8') if v is str else v for v in val])
    else:
        return val.encode('utf-8') if val is str else val


def _decode_utf8(val):
    if val is tuple:
        return tuple([v.decode('utf-8') if v is str else v for v in val])
    else:
        return val.decode('utf-8') if val is str else val


class Attribute(object):
    ''' An base attibute class.
    A attribute is an object who consist the attribute to describe another object. More specific, a
    attribute object only contains small descriptions, which can only be a tuple or a value/string (
    <64k) and can be stored in a hdf5 attibute.
    '''

    def __eq__(self, other):
        '''**Equality verify**
        :param other:
        :return: bool
        '''
        if self.__class__ != other.__class__:
            return False
        return self.__dict__ == other.__dict__

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
                group.attrs.create(key, data=_encode_utf8(value))

    @classmethod
    def load_h5(cls, path=None):
        if cls is Attribute:
            return NotImplementedError

        dict_attrs = cls().__dict__
        if path is None:
            path = 'tmp' + cls.__name__ + '.h5'
        with h5py.File(path, 'r') as fin:
            group = fin[cls.__name__]
            for key in dict_attrs.keys():
                dict_attrs[key] = _decode_utf8(group.attrs[key])
            return cls(**dict_attrs)

    def __str__(self):
        out_str = f'{type(self)} object at {hex(id(self))}\n'
        for key in self.__dict__.keys():
            out_str += f'{key}: {type(self.__dict__[key])} = {self.__dict__[key]}\n'
        return out_str

    @abstractmethod
    def map(self, _):
        raise NotImplementedError('map method is valid for ', self.__class__, ' object.')


class Object(object):
    _data: np.ndarray
    _attr: Attribute

    @property
    def data(self):
        return self._data

    @property
    def attr(self):
        return self._attr

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

        self.attr.save_h5(path, mode)
        with h5py.File(path, 'r+') as fout:
            fout.create_dataset('_data', data=self.data, compression="gzip")

    @classmethod
    def load_h5(cls, path=None):
        if cls is Object:
            return NotImplementedError

        attr = cls._attr.__class__.load_h5(path)
        with h5py.File(path, 'r') as fin:
            data = np.array(fin['_data'])
            return cls(data, attr)

    @abstractmethod
    def map(self, _):
        raise NotImplementedError

    def __str__(self):
        out_str = f'{type(self)} object at {hex(id(self))} with attributes as:\n'
        out_str += self.attr.__str__()
        return out_str

    def __neg__(self):
        def _neg(data):
            return -data

        return self.map(_neg)

    def __pos__(self):
        def _pos(data):
            return data

        return self.map(_pos)

    def __add__(self, other):
        def _add(o):
            def kernel(data):
                if isscalar(o) or isinstance(o, np.ndarray):
                    return data + o
                elif isinstance(o, self.__class__):
                    return data + o.data
                else:
                    raise NotImplementedError

            return kernel

        return self.map(_add(other))

    def __sub__(self, other):
        def _sub(o):
            def kernel(data):
                if isscalar(o) or isinstance(o, np.ndarray):
                    return data - o
                elif isinstance(o, self.__class__):
                    return data - o.data
                else:
                    raise NotImplementedError

            return kernel

        return self.map(_sub(other))

    def __mul__(self, other):
        def _mul(o):
            def kernel(data):
                if isscalar(o) or isinstance(o, np.ndarray):
                    return data * o
                elif isinstance(o, self.__class__):
                    return data * o.data
                else:
                    raise NotImplementedError

            return kernel

        return self.map(_mul(other))

    def __div__(self, other):
        def _div(o):
            def kernel(data):
                if isscalar(o) or isinstance(o, np.ndarray):
                    return data / o
                elif isinstance(o, self.__class__):
                    return data / o.data
                else:
                    raise NotImplementedError

            return kernel

        return self.map(_div(other))
