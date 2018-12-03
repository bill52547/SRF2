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

LARGE = 1e8
SMALL = 1e-8

__all__ = ('cuda_add_with_array', 'cuda_add_with_scale', 'cuda_sub_with_array',
           'cuda_sub_with_scale', 'cuda_mul_with_array', 'cuda_mul_with_scale',
           'cuda_truediv_with_array', 'cuda_truediv_with_scale',
           'cuda_iadd_with_array', 'cuda_iadd_with_scale', 'cuda_isub_with_array',
           'cuda_isub_with_scale', 'cuda_imul_with_array', 'cuda_imul_with_scale',
           'cuda_itruediv_with_array', 'cuda_itruediv_with_scale',)


@cuda.jit
def cuda_add_with_array(d_sum, d_array, d_other):
    tx, ty, tz = cuda.threadIdx.x, cuda.threadIdx.y, cuda.threadIdx.z
    bx, by, bz = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
    wx, wy, wz = cuda.blockDim.x, cuda.blockDim.y, cuda.blockDim.z
    x = tx + bx * wx
    y = ty + by * wy
    z = tz + bz * wz
    if x > d_array.shape[0] or y > d_array.shape[1] or z > d_array.shape[2]:
        return
    d_sum[x, y, z] = d_array[x, y, z] + d_other[x, y, z]


@cuda.jit
def cuda_add_with_scale(d_sum, d_array, scale):
    tx, ty, tz = cuda.threadIdx.x, cuda.threadIdx.y, cuda.threadIdx.z
    bx, by, bz = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
    wx, wy, wz = cuda.blockDim.x, cuda.blockDim.y, cuda.blockDim.z
    x = tx + bx * wx
    y = ty + by * wy
    z = tz + bz * wz
    if x > d_array.shape[0] or y > d_array.shape[1] or z > d_array.shape[2]:
        return
    d_sum[x, y, z] = d_array[x, y, z] + scale


@cuda.jit
def cuda_iadd_with_array(d_array, d_other):
    tx, ty, tz = cuda.threadIdx.x, cuda.threadIdx.y, cuda.threadIdx.z
    bx, by, bz = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
    wx, wy, wz = cuda.blockDim.x, cuda.blockDim.y, cuda.blockDim.z
    x = tx + bx * wx
    y = ty + by * wy
    z = tz + bz * wz
    if x > d_array.shape[0] or y > d_array.shape[1] or z > d_array.shape[2]:
        return
    d_array[x, y, z] += d_other[x, y, z]


@cuda.jit
def cuda_iadd_with_scale(d_array, scale):
    tx, ty, tz = cuda.threadIdx.x, cuda.threadIdx.y, cuda.threadIdx.z
    bx, by, bz = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
    wx, wy, wz = cuda.blockDim.x, cuda.blockDim.y, cuda.blockDim.z
    x = tx + bx * wx
    y = ty + by * wy
    z = tz + bz * wz
    if x > d_array.shape[0] or y > d_array.shape[1] or z > d_array.shape[2]:
        return
    d_array[x, y, z] += scale


@cuda.jit
def cuda_sub_with_array(d_dif, d_array, d_other):
    tx, ty, tz = cuda.threadIdx.x, cuda.threadIdx.y, cuda.threadIdx.z
    bx, by, bz = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
    wx, wy, wz = cuda.blockDim.x, cuda.blockDim.y, cuda.blockDim.z
    x = tx + bx * wx
    y = ty + by * wy
    z = tz + bz * wz
    if x > d_array.shape[0] or y > d_array.shape[1] or z > d_array.shape[2]:
        return
    d_dif[x, y, z] = d_array[x, y, z] - d_other[x, y, z]


@cuda.jit
def cuda_sub_with_scale(d_dif, d_array, scale):
    tx, ty, tz = cuda.threadIdx.x, cuda.threadIdx.y, cuda.threadIdx.z
    bx, by, bz = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
    wx, wy, wz = cuda.blockDim.x, cuda.blockDim.y, cuda.blockDim.z
    x = tx + bx * wx
    y = ty + by * wy
    z = tz + bz * wz
    if x > d_array.shape[0] or y > d_array.shape[1] or z > d_array.shape[2]:
        return
    d_dif[x, y, z] = d_array[x, y, z] - scale


@cuda.jit
def cuda_isub_with_array(d_array, d_other):
    tx, ty, tz = cuda.threadIdx.x, cuda.threadIdx.y, cuda.threadIdx.z
    bx, by, bz = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
    wx, wy, wz = cuda.blockDim.x, cuda.blockDim.y, cuda.blockDim.z
    x = tx + bx * wx
    y = ty + by * wy
    z = tz + bz * wz
    if x > d_array.shape[0] or y > d_array.shape[1] or z > d_array.shape[2]:
        return
    d_array[x, y, z] -= d_other[x, y, z]


@cuda.jit
def cuda_isub_with_scale(d_array, scale):
    tx, ty, tz = cuda.threadIdx.x, cuda.threadIdx.y, cuda.threadIdx.z
    bx, by, bz = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
    wx, wy, wz = cuda.blockDim.x, cuda.blockDim.y, cuda.blockDim.z
    x = tx + bx * wx
    y = ty + by * wy
    z = tz + bz * wz
    if x > d_array.shape[0] or y > d_array.shape[1] or z > d_array.shape[2]:
        return
    d_array[x, y, z] -= scale


@cuda.jit
def cuda_mul_with_array(d_prod, d_array, d_other):
    tx, ty, tz = cuda.threadIdx.x, cuda.threadIdx.y, cuda.threadIdx.z
    bx, by, bz = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
    wx, wy, wz = cuda.blockDim.x, cuda.blockDim.y, cuda.blockDim.z
    x = tx + bx * wx
    y = ty + by * wy
    z = tz + bz * wz
    if x > d_array.shape[0] or y > d_array.shape[1] or z > d_array.shape[2]:
        return
    d_prod[x, y, z] = d_array[x, y, z] * d_other[x, y, z]


@cuda.jit
def cuda_mul_with_scale(d_prod, d_array, scale):
    tx, ty, tz = cuda.threadIdx.x, cuda.threadIdx.y, cuda.threadIdx.z
    bx, by, bz = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
    wx, wy, wz = cuda.blockDim.x, cuda.blockDim.y, cuda.blockDim.z
    x = tx + bx * wx
    y = ty + by * wy
    z = tz + bz * wz
    if x > d_array.shape[0] or y > d_array.shape[1] or z > d_array.shape[2]:
        return
    d_prod[x, y, z] = d_array[x, y, z] * scale


@cuda.jit
def cuda_imul_with_array(d_array, d_other):
    tx, ty, tz = cuda.threadIdx.x, cuda.threadIdx.y, cuda.threadIdx.z
    bx, by, bz = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
    wx, wy, wz = cuda.blockDim.x, cuda.blockDim.y, cuda.blockDim.z
    x = tx + bx * wx
    y = ty + by * wy
    z = tz + bz * wz
    if x > d_array.shape[0] or y > d_array.shape[1] or z > d_array.shape[2]:
        return
    d_array[x, y, z] *= d_other[x, y, z]


@cuda.jit
def cuda_imul_with_scale(d_array, scale):
    tx, ty, tz = cuda.threadIdx.x, cuda.threadIdx.y, cuda.threadIdx.z
    bx, by, bz = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
    wx, wy, wz = cuda.blockDim.x, cuda.blockDim.y, cuda.blockDim.z
    x = tx + bx * wx
    y = ty + by * wy
    z = tz + bz * wz
    if x > d_array.shape[0] or y > d_array.shape[1] or z > d_array.shape[2]:
        return
    d_array[x, y, z] *= scale


@cuda.jit
def cuda_truediv_with_array(d_quo, d_array, d_other):
    tx, ty, tz = cuda.threadIdx.x, cuda.threadIdx.y, cuda.threadIdx.z
    bx, by, bz = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
    wx, wy, wz = cuda.blockDim.x, cuda.blockDim.y, cuda.blockDim.z
    x = tx + bx * wx
    y = ty + by * wy
    z = tz + bz * wz
    if x > d_array.shape[0] or y > d_array.shape[1] or z > d_array.shape[2]:
        return
    if abs(d_other[x, y, z]) < SMALL:
        d_quo[x, y, z] = LARGE
    else:
        d_quo[x, y, z] = d_array[x, y, z] / d_other[x, y, z]


@cuda.jit
def cuda_truediv_with_scale(d_quo, d_array, scale):
    tx, ty, tz = cuda.threadIdx.x, cuda.threadIdx.y, cuda.threadIdx.z
    bx, by, bz = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
    wx, wy, wz = cuda.blockDim.x, cuda.blockDim.y, cuda.blockDim.z
    x = tx + bx * wx
    y = ty + by * wy
    z = tz + bz * wz
    if x > d_array.shape[0] or y > d_array.shape[1] or z > d_array.shape[2]:
        return
    if abs(scale) < SMALL:
        d_quo[x, y, z] = LARGE
    else:
        d_quo[x, y, z] = d_array[x, y, z] / scale


@cuda.jit
def cuda_itruediv_with_array(d_array, d_other):
    tx, ty, tz = cuda.threadIdx.x, cuda.threadIdx.y, cuda.threadIdx.z
    bx, by, bz = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
    wx, wy, wz = cuda.blockDim.x, cuda.blockDim.y, cuda.blockDim.z
    x = tx + bx * wx
    y = ty + by * wy
    z = tz + bz * wz
    if x > d_array.shape[0] or y > d_array.shape[1] or z > d_array.shape[2]:
        return
    if abs(d_other[x, y, z]) < SMALL:
        d_array[x, y, z] = LARGE
    else:
        d_array[x, y, z] /= d_other[x, y, z]


@cuda.jit
def cuda_itruediv_with_scale(d_array, scale):
    tx, ty, tz = cuda.threadIdx.x, cuda.threadIdx.y, cuda.threadIdx.z
    bx, by, bz = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
    wx, wy, wz = cuda.blockDim.x, cuda.blockDim.y, cuda.blockDim.z
    x = tx + bx * wx
    y = ty + by * wy
    z = tz + bz * wz
    if x > d_array.shape[0] or y > d_array.shape[1] or z > d_array.shape[2]:
        return
    if abs(scale) < SMALL:
        d_array[x, y, z] = LARGE
    else:
        d_array[x, y, z] /= scale
