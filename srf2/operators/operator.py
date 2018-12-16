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

import sys
import inspect
import numpy as np


def _identity(f):
    return f


def _function_signature(func):
    """Return the signature of a callable as a string.

    Parameters
    ----------
    func : callable
        Function whose signature to extract.

    Returns
    -------
    sig : string
        Signature of the function.
    """
    if sys.version_info.major > 2:
        # Python 3 already implements this functionality
        return func.__name__ + str(inspect.signature(func))

    # In Python 2 we have to do it manually, unfortunately
    spec = inspect.getargspec(func)
    posargs = spec.args
    defaults = spec.defaults if spec.defaults is not None else []
    varargs = spec.varargs
    kwargs = spec.keywords
    deflen = 0 if defaults is None else len(defaults)
    nodeflen = 0 if posargs is None else len(posargs) - deflen

    args = ['{}'.format(arg) for arg in posargs[:nodeflen]]
    args.extend('{}={}'.format(arg, dval)
                for arg, dval in zip(posargs[nodeflen:], defaults))
    if varargs:
        args.append('*{}'.format(varargs))
    if kwargs:
        args.append('**{}'.format(kwargs))

    argstr = ', '.join(args)

    return '{}({})'.format(func.__name__, argstr)


class Operator:
    _op = _identity
    __platform_manager__ = 'cpu'


class Unary(Operator):
    def __init__(self, op = _identity):
        self._op = op


class Binocular(Operator):
    def __init__(self):
        pass
