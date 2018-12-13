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

BLOCK_DIM = 16

__all__ = ('cuda_iadd_with_array', 'cuda_iadd_with_scale',
           'cuda_isub_with_array', 'cuda_isub_with_scale',
           'cuda_imul_with_array', 'cuda_imul_with_scale',
           'cuda_itruediv_with_array', 'cuda_itruediv_with_scale',
           )


# iadd
def cuda_iadd_with_array(d_array, d_other, stream=0):
    blockdim = (BLOCK_DIM,) * d_array.ndim
    griddim = tuple((x - 1) // BLOCK_DIM + 1 for x in d_array.shape)
    if d_array.ndim == 1:
        kernel_iadd_with_array_1d[griddim, blockdim](d_array, d_other)
    elif d_array.ndim == 2:
        kernel_iadd_with_array_2d[griddim, blockdim](d_array, d_other)
    elif d_array.ndim == 3:
        kernel_iadd_with_array_3d[griddim, blockdim](d_array, d_other)
    else:
        raise NotImplementedError


@cuda.jit
def kernel_iadd_with_array_1d(d_array, d_other):
    x = cuda.grid(1)
    if not (x < d_array.shape[0]):
        return
    d_array[x] += d_other[x]


@cuda.jit
def kernel_iadd_with_array_2d(d_array, d_other):
    x, y = cuda.grid(2)
    if not (x < d_array.shape[0] and y < d_array.shape[1]):
        return
    d_array[x, y] += d_other[x, y]


@cuda.jit
def kernel_iadd_with_array_3d(d_array, d_other):
    x, y, z = cuda.grid(3)
    if not (x < d_array.shape[0] and y < d_array.shape[1] and y < d_array.shape[2]):
        return
    d_array[x, y, z] += d_other[x, y, z]


def cuda_iadd_with_scale(d_array, scale, stream=0):
    blockdim = (BLOCK_DIM,) * d_array.ndim
    griddim = tuple((x - 1) // BLOCK_DIM + 1 for x in d_array.shape)
    if d_array.ndim == 1:
        kernel_iadd_with_scale_1d[griddim, blockdim](d_array, scale)
    elif d_array.ndim == 2:
        kernel_iadd_with_scale_2d[griddim, blockdim](d_array, scale)
    elif d_array.ndim == 3:
        kernel_iadd_with_scale_3d[griddim, blockdim](d_array, scale)
    else:
        raise NotImplementedError


@cuda.jit
def kernel_iadd_with_scale_1d(d_array, scale):
    x = cuda.grid(1)
    if not (x < d_array.shape[0]):
        return
    d_array[x] += scale


@cuda.jit
def kernel_iadd_with_scale_2d(d_array, scale):
    x, y = cuda.grid(2)
    if not (x < d_array.shape[0] and y < d_array.shape[1]):
        return
    d_array[x, y] += scale


@cuda.jit
def kernel_iadd_with_scale_3d(d_array, scale):
    x, y, z = cuda.grid(3)
    if not (x < d_array.shape[0] and y < d_array.shape[1] and y < d_array.shape[2]):
        return
    d_array[x, y, z] += scale

# isub
def cuda_isub_with_array(d_array, d_other, stream=0):
    blockdim = (BLOCK_DIM,) * d_array.ndim
    griddim = tuple((x - 1) // BLOCK_DIM + 1 for x in d_array.shape)
    if d_array.ndim == 1:
        kernel_isub_with_array_1d[griddim, blockdim](d_array, d_other)
    elif d_array.ndim == 2:
        kernel_isub_with_array_2d[griddim, blockdim](d_array, d_other)
    elif d_array.ndim == 3:
        kernel_isub_with_array_3d[griddim, blockdim](d_array, d_other)
    else:
        raise NotImplementedError


@cuda.jit
def kernel_isub_with_array_1d(d_array, d_other):
    x = cuda.grid(1)
    if not (x < d_array.shape[0]):
        return
    d_array[x] -= d_other[x]


@cuda.jit
def kernel_isub_with_array_2d(d_array, d_other):
    x, y = cuda.grid(2)
    if not (x < d_array.shape[0] and y < d_array.shape[1]):
        return
    d_array[x, y] -= d_other[x, y]


@cuda.jit
def kernel_isub_with_array_3d(d_array, d_other):
    x, y, z = cuda.grid(3)
    if not (x < d_array.shape[0] and y < d_array.shape[1] and y < d_array.shape[2]):
        return
    d_array[x, y, z] -= d_other[x, y, z]


def cuda_isub_with_scale(d_array, scale, stream=0):
    blockdim = (BLOCK_DIM,) * d_array.ndim
    griddim = tuple((x - 1) // BLOCK_DIM + 1 for x in d_array.shape)
    if d_array.ndim == 1:
        kernel_isub_with_scale_1d[griddim, blockdim](d_array, scale)
    elif d_array.ndim == 2:
        kernel_isub_with_scale_2d[griddim, blockdim](d_array, scale)
    elif d_array.ndim == 3:
        kernel_isub_with_scale_3d[griddim, blockdim](d_array, scale)
    else:
        raise NotImplementedError


@cuda.jit
def kernel_isub_with_scale_1d(d_array, scale):
    x = cuda.grid(1)
    if not (x < d_array.shape[0]):
        return
    d_array[x] -= scale


@cuda.jit
def kernel_isub_with_scale_2d(d_array, scale):
    x, y = cuda.grid(2)
    if not (x < d_array.shape[0] and y < d_array.shape[1]):
        return
    d_array[x, y] -= scale


@cuda.jit
def kernel_isub_with_scale_3d(d_array, scale):
    x, y, z = cuda.grid(3)
    if not (x < d_array.shape[0] and y < d_array.shape[1] and y < d_array.shape[2]):
        return
    d_array[x, y, z] -= scale


# imul
def cuda_imul_with_array(d_array, d_other, stream=0):
    blockdim = (BLOCK_DIM,) * d_array.ndim
    griddim = tuple((x - 1) // BLOCK_DIM + 1 for x in d_array.shape)
    if d_array.ndim == 1:
        kernel_imul_with_array_1d[griddim, blockdim](d_array, d_other)
    elif d_array.ndim == 2:
        kernel_imul_with_array_2d[griddim, blockdim](d_array, d_other)
    elif d_array.ndim == 3:
        kernel_imul_with_array_3d[griddim, blockdim](d_array, d_other)
    else:
        raise NotImplementedError


@cuda.jit
def kernel_imul_with_array_1d(d_array, d_other):
    x = cuda.grid(1)
    if not (x < d_array.shape[0]):
        return
    d_array[x] *= d_other[x]


@cuda.jit
def kernel_imul_with_array_2d(d_array, d_other):
    x, y = cuda.grid(2)
    if not (x < d_array.shape[0] and y < d_array.shape[1]):
        return
    d_array[x, y] *= d_other[x, y]


@cuda.jit
def kernel_imul_with_array_3d(d_array, d_other):
    x, y, z = cuda.grid(3)
    if not (x < d_array.shape[0] and y < d_array.shape[1] and y < d_array.shape[2]):
        return
    d_array[x, y, z] *= d_other[x, y, z]


def cuda_imul_with_scale(d_array, scale, stream=0):
    blockdim = (BLOCK_DIM,) * d_array.ndim
    griddim = tuple((x - 1) // BLOCK_DIM + 1 for x in d_array.shape)
    if d_array.ndim == 1:
        kernel_imul_with_scale_1d[griddim, blockdim](d_array, scale)
    elif d_array.ndim == 2:
        kernel_imul_with_scale_2d[griddim, blockdim](d_array, scale)
    elif d_array.ndim == 3:
        kernel_imul_with_scale_3d[griddim, blockdim](d_array, scale)
    else:
        raise NotImplementedError


@cuda.jit
def kernel_imul_with_scale_1d(d_array, scale):
    x = cuda.grid(1)
    if not (x < d_array.shape[0]):
        return
    d_array[x] *= scale


@cuda.jit
def kernel_imul_with_scale_2d(d_array, scale):
    x, y = cuda.grid(2)
    if not (x < d_array.shape[0] and y < d_array.shape[1]):
        return
    d_array[x, y] *= scale


@cuda.jit
def kernel_imul_with_scale_3d(d_array, scale):
    x, y, z = cuda.grid(3)
    if not (x < d_array.shape[0] and y < d_array.shape[1] and y < d_array.shape[2]):
        return
    d_array[x, y, z] *= scale


# itruediv
def cuda_itruediv_with_array(d_array, d_other, stream=0):
    blockdim = (BLOCK_DIM,) * d_array.ndim
    griddim = tuple((x - 1) // BLOCK_DIM + 1 for x in d_array.shape)
    if d_array.ndim == 1:
        kernel_itruediv_with_array_1d[griddim, blockdim](d_array, d_other)
    elif d_array.ndim == 2:
        kernel_itruediv_with_array_2d[griddim, blockdim](d_array, d_other)
    elif d_array.ndim == 3:
        kernel_itruediv_with_array_3d[griddim, blockdim](d_array, d_other)
    else:
        raise NotImplementedError


@cuda.jit
def kernel_itruediv_with_array_1d(d_array, d_other):
    x = cuda.grid(1)
    if not (x < d_array.shape[0]):
        return
    d_array[x] /= d_other[x]


@cuda.jit
def kernel_itruediv_with_array_2d(d_array, d_other):
    x, y = cuda.grid(2)
    if not (x < d_array.shape[0] and y < d_array.shape[1]):
        return
    d_array[x, y] /= d_other[x, y]


@cuda.jit
def kernel_itruediv_with_array_3d(d_array, d_other):
    x, y, z = cuda.grid(3)
    if not (x < d_array.shape[0] and y < d_array.shape[1] and y < d_array.shape[2]):
        return
    d_array[x, y, z] /= d_other[x, y, z]


def cuda_itruediv_with_scale(d_array, scale, stream=0):
    blockdim = (BLOCK_DIM,) * d_array.ndim
    griddim = tuple((x - 1) // BLOCK_DIM + 1 for x in d_array.shape)
    if d_array.ndim == 1:
        kernel_itruediv_with_scale_1d[griddim, blockdim](d_array, scale)
    elif d_array.ndim == 2:
        kernel_itruediv_with_scale_2d[griddim, blockdim](d_array, scale)
    elif d_array.ndim == 3:
        kernel_itruediv_with_scale_3d[griddim, blockdim](d_array, scale)
    else:
        raise NotImplementedError


@cuda.jit
def kernel_itruediv_with_scale_1d(d_array, scale):
    x = cuda.grid(1)
    if not (x < d_array.shape[0]):
        return
    d_array[x] /= scale


@cuda.jit
def kernel_itruediv_with_scale_2d(d_array, scale):
    x, y = cuda.grid(2)
    if not (x < d_array.shape[0] and y < d_array.shape[1]):
        return
    d_array[x, y] /= scale


@cuda.jit
def kernel_itruediv_with_scale_3d(d_array, scale):
    x, y, z = cuda.grid(3)
    if not (x < d_array.shape[0] and y < d_array.shape[1] and y < d_array.shape[2]):
        return
    d_array[x, y, z] /= scale
