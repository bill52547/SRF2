#!/usr/bin/env python
# encoding: utf-8
'''
@author: Minghao Guo, Xiang Hong, Gaoyu Chen and Weijie Tao
@license: LGPL_v3.0
@contact: mh.guo0111@gmail.com
@software: srf_v2
@file: setup.py
@date: 11/10/2018
@desc: new version of Scalable Reconstruction Framework for Medical Imaging
'''

__all__ = ()
#
from .image_attr import *
from .projection_attr import *

#
__all__ += image_attr.__all__
__all__ += projection_attr.__all__
