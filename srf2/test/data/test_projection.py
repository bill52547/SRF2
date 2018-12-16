# encoding: utf-8
'''
@author: Minghao Guo
@license: MIT
@contact: mh.guo0111@gmail.com
@software: personal
@file: test_projection.py
@date: 11/29/2018
@desc: for personal usage
'''

import numpy as np

from srf2.attr.projection_attr import *
from srf2.data.projection import *


class Test_Projection:
    def test_init(self):
        param = {'shape': (1, 1), 'center': (0, 0), 'size': (1, 1), 'dims': ('u', 'v')}
        d = DetectorAttr(**param)
        p = ProjectionAttr(200, 100, 0, d)
        assert not Projection(p)
        Projection(p, np.zeros((1, 1)))

    def test_property(self):
        pass

    def test_bool(self):
        pass

    def test_map(self):
        pass

    def test_eq(self):
        pass

    def test_transpose(self):
        pass

    def test_squeeze(self):
        param = {'shape': (2, 2), 'center': (0, 0), 'size': (1, 1), 'dims': ('u', 'v')}
        d = DetectorAttr(**param)
        p = ProjectionAttr(200, 100, 0, d)
        proj = Projection(p, np.random.random(d.shape))
        assert proj == proj.squeeze()

        param = {'shape': (2, 1), 'center': (0, 0), 'size': (1, 1), 'dims': ('u', 'v')}
        d = DetectorAttr(**param)
        p = ProjectionAttr(200, 100, 0, d)
        proj = Projection(p, np.random.random(d.shape))

        param1 = {'shape': (2,), 'center': (0,), 'size': (1,), 'dims': ('u',)}
        d1 = DetectorAttr(**param1)
        p1 = ProjectionAttr(200, 100, 0, d1).squeeze()
        proj1 = Projection(p1, proj.data.squeeze())
        assert proj1 == proj.squeeze()


class Test_ProjectionCurve:
    pass


class Test_ProjectionFlat:
    pass


class Test_ProjectionSeries:
    def test_init(self):
        pass

#
#
# class Test_Image0D:
#     pass
#
#
# class Test_Image1D:
#     pass
#
#
# class Test_Image2D:
#     pass
#
#
# class Test_Image3D:
#     pass
# #       ass
# #     def test_property(self):
# #         data = np.random.random((2, 3, 4))
# #         meta = Image_meta(data.shape)
# #         image = Image(data)
# #
# #         assert meta == image.meta
# #         assert np.array_equal(data, image.data)
# #
# #     def test_io(self):
# #         data = np.random.random((2, 3, 4))
# #         image = Image(data)
# #         image.save_h5()
# #         image2 = Image.load_h5()
# #         assert image == image2
# #
# #     def test_slice(self):
# #         data = np.random.random((2, 3, 4))
# #         meta = Image_meta((2, 3, 4), (0, 0, 0), (2, 3, 4))
# #         image = Image(data, meta)
# #         image2 = Image(data[:, :, 1:-1], Image_meta((2, 3, 2), (0, 0, 0), (2, 3, 2)))
# #
# #         assert image[:, :, 1:-1] == image2
# #
# #
# # class Test_image_2d:
# #     pass
# #
# #
# # class Test_image_3d:
# #     def test_transpose(self):
# #         data = np.random.random((2, 3, 4))
# #         meta = Image_meta_3d(data.shape)
# #         image = Image_3d(data, meta)
# #         assert np.array_equal(image.data.transpose([1, 2, 0]), image.transpose([1, 2, 0]).data)
# #         assert np.array_equal(image.data.transpose([1, 2, 0]),
# #                               image.transpose(['y', 'z', 'x']).data)
