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

import h5py

# dt = h5py.special_dtype(vlen = str)
__all__ = ('Meta', 'Singleton',)


def _str_to_ascii(str):
    return list(map(ord, str))


def _ascii_to_str(num):
    from functools import reduce
    return reduce(lambda x, y: x + y, list(map(chr, num)))


class Meta(object):
    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        return self.__dict__ == other.__dict__

    def save_h5(self, path = None, mode = 'w'):
        group_name = self.__class__.__name__
        attrs_dict = self.__dict__
        if path is None:
            path = 'tmp' + group_name + '.h5'
        with h5py.File(path, mode) as fout:
            group = fout.create_group(group_name)
            for key, value in attrs_dict.items():
                if isinstance(value[0], str):
                    value = [12300111] + _str_to_ascii(value)
                group.attrs.create(key, data = value)

    @classmethod
    def load_h5(cls, path = None):
        instance = cls()
        group_name = instance.__class__.__name__
        attrs_dict = instance.__dict__
        if path is None:
            path = 'tmp' + group_name + '.h5'
        with h5py.File(path, 'r') as fin:
            group = fin[group_name]
            args = ()
            for key in attrs_dict.keys():
                tmp = group.attrs[key]
                if tmp[0] == 12300111:
                    tmp = _ascii_to_str(tmp[1:])
                args += tuple(tmp),
            print(args)
            return cls(*args)


class Singleton(object):
    instance = None

    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

# TODO: only allow one-char input now
