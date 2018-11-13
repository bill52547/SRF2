import numpy as np

from srf2.meta.image_meta import *


class Test_image_meta:
    def test_init(self):
        shape = (1, 1, 1)
        center = (0, 0, 0)
        size = (1, 1, 1)
        dims = ('x', 'y', 'z')

        assert Image_meta() == Image_meta(shape)
        assert Image_meta() == Image_meta(shape, center)
        assert Image_meta() == Image_meta(shape, center, size)
        assert Image_meta() == Image_meta(shape, center, size, dims)

    def test_property(self):
        shape = (1, 1, 1)
        center = (0, 0, 0)
        size = (1, 1, 1)
        dims = ('x', 'y', 'z')

        assert Image_meta().shape == Image_meta()._shape == shape
        assert Image_meta().center == Image_meta()._center == center
        assert Image_meta().size == Image_meta()._size == size
        assert Image_meta().dims == Image_meta()._dims == dims
        assert Image_meta().unit_size == (1, 1, 1)
        assert Image_meta().ndim == 3
        assert Image_meta().n_all == 1

        shape = (1, 2)
        center = (0, 0)
        size = (1, 1)
        dims = ('x', 'z')
        meta = Image_meta_2d(shape, center, size, dims)

        assert meta.n_x == meta.n_y == meta.n_t == 1
        assert meta.n_z == 2

    def test_slice(self):
        assert Image_meta() == Image_meta()[:, :, :]

        shape = (1, 2, 3)
        center = (0, 0, 4.5)
        size = (7, 8, 9)
        dims = ('x', 'y', 'z')

        assert Image_meta((1, 2, 1), (0, 0, 1.5), (7, 8, 3)) == Image_meta(shape, center,
                                                                           size)[:, :, 0:1]

    def test_locate(self):
        assert Image_meta().locate((0, 0, 0)) == (0, 0, 0)
        assert Image_meta().locate((0, 0, 0.5)) == (0, 0, 0.5)
        assert Image_meta().locate((-1, 0, 0.5)) == (-1, 0, 0.5)

    def test_transpose(self):
        shape = (1, 2, 3)
        dims2 = ('z', 'y', 'x')
        dims3 = ('z', 'x', 'y')

        assert Image_meta(shape).transpose() == Image_meta((3, 2, 1), (0, 0, 0), (1, 1, 1), dims2)
        assert Image_meta(shape).transpose([2, 0, 1]) == Image_meta((3, 1, 2), (0, 0, 0), (1, 1, 1),
                                                                    dims3)
        assert Image_meta(shape).transpose([2, 0, 1]) == Image_meta(shape).transpose(
            ['z', 'x', 'y'])


class Test_image_meta_singleton:
    meta = Image_meta_singleton()


class Test_image_meta_2d:
    def test_init(self):
        shape = (1, 1)
        center = (0, 0)
        size = (1, 1)
        dims = ('x', 'y')

        assert Image_meta_2d() == Image_meta_2d(shape)
        assert Image_meta_2d() == Image_meta_2d(shape, center)
        assert Image_meta_2d() == Image_meta_2d(shape, center, size)
        assert Image_meta_2d() == Image_meta_2d(shape, center, size, dims)

    def test_meshgrid(self):
        meta = Image_meta_2d((3, 4))
        assert meta.meshgrid()[0].shape == meta.meshgrid()[1].shape == meta.shape

    def test_theta(self):
        meta = Image_meta_2d((2, 2))
        assert np.array_equal(meta.theta(), np.pi / 4 * np.array([[-3, 3], [-1, 1]]))


class Test_image_meta_2d_singleton:
    meta = Image_meta_2d_singleton()


class Test_image_meta_3d:
    def test_init(self):
        shape = (1, 1, 1)
        center = (0, 0, 0)
        size = (1, 1, 1)
        dims = ('x', 'y', 'z')

        assert Image_meta_3d() == Image_meta_3d(shape)
        assert Image_meta_3d() == Image_meta_3d(shape, center)
        assert Image_meta_3d() == Image_meta_3d(shape, center, size)
        assert Image_meta_3d() == Image_meta_3d(shape, center, size, dims)

    def test_meshgrid(self):
        meta = Image_meta_3d((3, 4, 5))
        assert meta.meshgrid()[0].shape == meta.meshgrid()[1].shape == meta.meshgrid()[
            2].shape == meta.shape

        meta = Image_meta_3d((3, 4, 5))
        assert meta.meshgrid([slice(0, 3, 1), slice(0, 4, 1), slice(1, 3, None)])[0].shape == (
            3, 4, 2)


class Test_image_meta_3d_singleton:
    meta = Image_meta_3d_singleton()


class Test_image_meta_2d_3d_singleton():
    meta2 = Image_meta_2d_singleton()
    meta3 = Image_meta_3d_singleton()
    assert id(meta2) != id(meta3)
