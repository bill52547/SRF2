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
from numpy.core import isscalar

__all__ = ('Attribute', 'ObjectWithAttrData')


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
    _data = None

    def __init__(self, attr, data):
        self._attr = attr
        self._data = data

    @property
    def attr(self):
        return self._attr

    @property
    def data(self):
        return self._data

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        if self.attr != other.attr:
            return False

        if not self and not other:
            return True
        elif not self or not other:
            return False

        return np.array_equal(self.data, other.data)

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
            return cls(data, attr)

    @abstractmethod
    def map(self, _):
        raise NotImplementedError

    def __repr__(self):
        out_str = '{0} object at {1} with attributes as: '.format(type(self), hex(id(self)),
                                                                  end = '\n')
        out_str += self.attr.__repr__()
        return out_str

    def __neg__(self):
        def _neg(data, attr):
            return -data, attr

        return self.map(_neg)

    def __pos__(self):
        def _pos(data, attr):
            return data, attr

        return self.map(_pos)

    def __add__(self, other):
        def _add(o):
            def kernel(data, attr):
                if isscalar(o) or isinstance(o, np.ndarray):
                    return data + o, attr
                elif isinstance(o, self.__class__):
                    return data + o.data, attr
                else:
                    raise NotImplementedError

            return kernel

        return self.map(_add(other))

    __radd__ = __add__

    def __iadd__(self, other):
        if isscalar(other) or isinstance(other, np.ndarray):
            self._data += other
        elif isinstance(other, self.__class__):
            if self.attr == other.attr:
                self._data += other.data
            else:
                raise ValueError
        else:
            raise ValueError

    def __sub__(self, other):
        def _sub(o):
            def kernel(data, attr):
                if isscalar(o) or isinstance(o, np.ndarray):
                    return data - o, attr
                elif isinstance(o, self.__class__):
                    return data - o.data, attr
                else:
                    raise NotImplementedError

            return kernel

        return self.map(_sub(other))

    def __rsub__(self, other):
        def _rsub(o):
            def kernel(data, attr):
                if isscalar(o) or isinstance(o, np.ndarray):
                    return o - data, attr
                elif isinstance(o, self.__class__):
                    return o.data - data, attr
                else:
                    raise ValueError

            return kernel

        return self.map(_rsub(other))

    def __isub__(self, other):
        if isscalar(other) or isinstance(other, np.ndarray):
            self._data -= other
        elif isinstance(other, self.__class__):
            if self.attr == other.attr:
                self._data -= other.data
            else:
                raise ValueError
        else:
            raise ValueError


    def __mul__(self, other):
        def _mul(o):
            def kernel(data, attr):
                if isscalar(o) or isinstance(o, np.ndarray):
                    return data * o, attr
                elif isinstance(o, self.__class__):
                    return data * o.data, attr
                else:
                    raise NotImplementedError

            return kernel

        return self.map(_mul(other))

    __rmul__ = __mul__

    def __imul__(self, other):
        if isscalar(other) or isinstance(other, np.ndarray):
            self._data *= other
        elif isinstance(other, self.__class__):
            if self.attr == other.attr:
                self._data *= other.data
            else:
                raise ValueError
        else:
            raise ValueError

    def __truediv__(self, other):
        def _truediv(o):
            def kernel(data, attr):
                if isscalar(o) or isinstance(o, np.ndarray):
                    return data / o, attr
                elif isinstance(o, self.__class__):
                    return data / o.data, attr
                else:
                    raise NotImplementedError

            return kernel

        return self.map(_truediv(other))
