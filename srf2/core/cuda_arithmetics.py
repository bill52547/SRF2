#!/usr/bin/env python
# encoding: utf-8
'''
@author: Minghao Guo
@license: MIT
@contact: mh.guo0111@gmail.com
@software: personal
@file: Arithmetics.py
@date: 12/1/2018
@desc: for personal usage
'''

from numba import cuda
import numpy as np

BLOCK_DIM = 16

__all__ = ('cuda_iadd_with_array', 'cuda_iadd_with_scale',
           'cuda_isub_with_array', 'cuda_isub_with_scale',
           'cuda_imul_with_array', 'cuda_imul_with_scale',
           'cuda_itruediv_with_array', 'cuda_itruediv_with_scale',
           )


# iadd
def cuda_iadd_with_array(d_array, d_other, stream=0):
    blockdim = (BLOCK_DIM,) * len(d_array.shape)
    griddim = tuple(x // y + 1 for (x, y) in zip(d_array.shape, blockdim))
    if 1 <= len(d_array.shape) <= 3:
        kernel_iadd_with_array[griddim, blockdim](d_array, d_other)
    else:
        raise NotImplementedError


@cuda.jit
def kernel_iadd_with_array(d_array, d_other):
    ind = cuda.grid(len(d_array.shape))
    if not (np.array(ind) < np.array(d_array.shape)).all():
        return
    d_array[ind] += d_other[ind]


def cuda_iadd_with_scale(d_array, scale, stream=0):
    blockdim = (BLOCK_DIM,) * len(d_array.shape)
    griddim = tuple(x // y + 1 for (x, y) in zip(d_array.shape, blockdim))
    if 1 <= len(d_array.shape) <= 3:
        kernel_iadd_with_scale[griddim, blockdim](d_array, scale)
    else:
        raise NotImplementedError


@cuda.jit
def kernel_iadd_with_scale(d_array, scale):
    ind = cuda.grid(len(d_array.shape))
    if not (np.array(ind) < np.array(d_array.shape)).all():
        return
    d_array[ind] += scale


# isub
def cuda_isub_with_array(d_array, d_other, stream=0):
    blockdim = (BLOCK_DIM,) * len(d_array.shape)
    griddim = tuple(x // y + 1 for (x, y) in zip(d_array.shape, blockdim))
    if 1 <= len(d_array.shape) <= 3:
        kernel_isub_with_array[griddim, blockdim](d_array, d_other)
    else:
        raise NotImplementedError


@cuda.jit
def kernel_isub_with_array(d_array, d_other):
    ind = cuda.grid(len(d_array.shape))
    if not (np.array(ind) < np.array(d_array.shape)).all():
        return
    d_array[ind] -= d_other[ind]


def cuda_isub_with_scale(d_array, scale, stream=0):
    blockdim = (BLOCK_DIM,) * len(d_array.shape)
    griddim = tuple(x // y + 1 for (x, y) in zip(d_array.shape, blockdim))
    if 1 <= len(d_array.shape) <= 3:
        kernel_isub_with_scale[griddim, blockdim](d_array, scale)
    else:
        raise NotImplementedError


@cuda.jit
def kernel_isub_with_scale(d_array, scale):
    ind = cuda.grid(len(d_array.shape))
    if not (np.array(ind) < np.array(d_array.shape)).all():
        return
    d_array[ind] -= scale


# imul
def cuda_imul_with_array(d_array, d_other, stream=0):
    blockdim = (BLOCK_DIM,) * len(d_array.shape)
    griddim = tuple(x // y + 1 for (x, y) in zip(d_array.shape, blockdim))
    if 1 <= len(d_array.shape) <= 3:
        kernel_imul_with_array[griddim, blockdim](d_array, d_other)
    else:
        raise NotImplementedError


@cuda.jit
def kernel_imul_with_array(d_array, d_other):
    ind = cuda.grid(len(d_array.shape))
    if not (np.array(ind) < np.array(d_array.shape)).all():
        return
    d_array[ind] *= d_other[ind]


def cuda_imul_with_scale(d_array, scale, stream=0):
    blockdim = (BLOCK_DIM,) * len(d_array.shape)
    griddim = tuple(x // y + 1 for (x, y) in zip(d_array.shape, blockdim))
    if 1 <= len(d_array.shape) <= 3:
        kernel_itruediv_with_scale[griddim, blockdim](d_array, scale)
    else:
        raise NotImplementedError


@cuda.jit
def kernel_imul_with_scale(d_array, scale):
    ind = cuda.grid(len(d_array.shape))
    if not (np.array(ind) < np.array(d_array.shape)).all():
        return
    d_array[ind] *= scale


# itruediv
def cuda_itruediv_with_array(d_array, d_other, stream=0):
    blockdim = (BLOCK_DIM,) * len(d_array.shape)
    griddim = tuple(x // y + 1 for (x, y) in zip(d_array.shape, blockdim))
    if 1 <= len(d_array.shape) <= 3:
        kernel_itruediv_with_array[griddim, blockdim](d_array, d_other)
    else:
        raise NotImplementedError


@cuda.jit
def kernel_itruediv_with_array(d_array, d_other):
    ind = cuda.grid(len(d_array.shape))
    if not (np.array(ind) < np.array(d_array.shape)).all():
        return
    d_array[ind] /= d_other[ind]


def cuda_itruediv_with_scale(d_array, scale, stream=0):
    blockdim = (BLOCK_DIM,) * len(d_array.shape)
    griddim = tuple(x // y + 1 for (x, y) in zip(d_array.shape, blockdim))
    if 1 <= len(d_array.shape) <= 3:
        kernel_itruediv_with_scale[griddim, blockdim](d_array, scale)
    else:
        raise NotImplementedError


@cuda.jit
def kernel_itruediv_with_scale(d_array, scale):
    ind = cuda.grid(len(d_array.shape))
    if not (np.array(ind) < np.array(d_array.shape)).all():
        return
    d_array[ind] /= scale
