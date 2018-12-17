# encoding: utf-8
'''
@author: Minghao Guo, Xiang Hong, Gaoyu Chen and Weijie Tao
@license: LGPL_v3.0
@contact: mh.guo0111@gmail.com
@software: srf_v2
@file: test_utils.py
@date: 12/17/2018
@desc: new version of Scalable Reconstruction Framework for Medical Imaging
'''

import pytest
from numba import cuda

__all__ = ('ensure_gpu',)

'''skip if no gpu or cuda installed. used as decorator pf pytest unit'''
ensure_gpu = pytest.mark.skipif(not cuda.is_available(), reason = 'No supported GPU is installed')
