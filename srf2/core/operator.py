#!/usr/bin/env python
# encoding: utf-8
'''
@author: Minghao Guo, Xiang Hong, Gaoyu Chen and Weijie Tao
@license: LGPL_v3.0
@contact: mh.guo0111@gmail.com
@software: srf_v2
@file: operator.py
@date: 12/13/18
@desc: new version of Scalable Reconstruction Framework for Medical Imaging
'''

__all__ = ('Operator',)


def _default_call_out_of_place(op, *args, **kwargs):
    out = new(op.out_domain)
    result = op._call_in_place(*args, out, **kwargs)
    return out, result


def _default_call_in_place(op, *args, out = None, **kwargs):
    if not isinstance(out, op.out_domain):
        raise TypeError('`out` is not a `{}` instance'.format(op.out_domain))
    out2, result = op._call_out_of_place(*args, **kwargs)
    out = out2
    return result


class Operator:
    __call_mode__ = None
    __device_manager__ = 'host'

    def __new__(cls, *args, **kwargs):
        for parent in cls.mro():
            if callable(parent):
                cls.__call__ = parent.__call__
                cls.__call_mode__ = parent.__call_mode__
                cls.__device_manager__ = parent.__device_manager__
                return parent.__new__(cls)
        return object.__new__(cls)

    def __init__(self, in_domain, out_domain):
        self._in_domain = in_domain
        self._out_domain = out_domain

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def in_domain(self):
        return self._in_domain

    @property
    def out_domain(self):
        return self._out_domain

    @property
    def call_mode(self):
        return self.__call_mode__

    @property
    def device_manager(self):
        return self.__device_manager__

    @property
    def adjoint(self):
        raise NotImplementedError
