import numpy as np

from srf2.attr.image_attr import *


class Test_ImageAttr:
    def test_init(self):
        shape = (1, 1, 1)
        center = (0, 0, 0)
        size = (1, 1, 1)
        dims = ('x', 'y', 'z')
        assert ImageAttr(shape, center, size, dims) == ImageAttr(shape) \
               == ImageAttr(shape, center) == ImageAttr(shape, center, size)

    def test_property(self):
        shape = (1, 2, 3)
        center = (3, 2, 1)
        size = (1, 2, 3)
        dims = ('x', 'y', 'z')
        attr = ImageAttr(shape, center, size, dims)

        assert attr.shape == attr._shape == shape
        assert attr.center == attr._center == center
        assert attr.size == attr._size == size
        assert attr.dims == attr._dims == dims
        assert attr.unit_size == (1, 1, 1)
        assert attr.ndim == 3
        assert attr.n_x == 1
        assert attr.n_y == 2
        assert attr.n_z == 3
        assert attr.n_t == 1
        assert attr.num() == 6

    def test_getitem(self):
        shape = (4, 2, 3)
        center = (0, 0, 0)
        size = (4, 2, 3)
        dims = ('x', 'y', 'z')
        attr = ImageAttr(shape, center, size, dims)
        assert isinstance(attr[:, :, :], ImageAttr)

        shape = (1, 2, 3)
        center = (0, 0, 4.5)
        size = (7, 8, 9)
        dims = ('x', 'y', 'z')

        assert ImageAttr((1, 2, 1), (0, 0, 1.5), (7, 8, 3), ('x', 'y', 'z')) == \
               ImageAttr(shape, center, size, dims)[:, :, 0:1]

    def test_squeeze(self):
        shape = (4, 1, 2)
        center = (0, 0, 0)
        size = (4, 1, 2)
        dims = ('x', 'y', 'z')
        attr = ImageAttr(shape, center, size, dims)
        shape = (4, 2)
        center = (0, 0)
        size = (4, 2)
        dims = ('x', 'z')
        attr2 = Image2DAttr(shape, center, size, dims)
        assert attr.squeeze() == attr2

    def test_locate(self):
        shape = (1, 1, 1)
        assert ImageAttr(shape).locate((0, 0, 0)) == (0, 0, 0)
        assert ImageAttr(shape).locate((0, 0, 0.5)) == (0, 0, 0.5)
        assert ImageAttr(shape).locate((-1, 0, 0.5)) == (-1, 0, 0.5)

    def test_transpose(self):
        shape = (1, 2, 3)
        center = (0, 0, 0)
        size = (1, 2, 3)
        dims = ('x', 'y', 'z')
        attr = ImageAttr(shape, center, size, dims)
        shape2 = (3, 1, 2)
        center2 = (0, 0, 0)
        size2 = (3, 1, 2)
        dims2 = ('z', 'x', 'y')
        attr2 = ImageAttr(shape2, center2, size2, dims2)

        shape3 = (3, 2, 1)
        center3 = (0, 0, 0)
        size3 = (3, 2, 1)
        dims3 = ('z', 'y', 'x')
        attr3 = ImageAttr(shape3, center3, size3, dims3)

        assert attr.transpose([2, 0, 1]) == attr2
        assert attr.transpose(dims2) == attr2
        assert attr.transpose() == attr3 == attr.T


class Test_Image0DAttr:
    def test_init(self):
        shape = center = size = dims = tuple([])
        Image0DAttr(shape, center, size, dims)


class Test_Image1DAttr:
    def test_init(self):
        shape = (3,)
        center = (0,)
        size = (3,)
        dims = ('x',)
        assert Image1DAttr(shape) == Image1DAttr(shape, center) == \
               Image1DAttr(shape, center, size) == Image1DAttr(shape, center, size, dims)

    def test_getitem(self):
        attr = Image1DAttr((5,), (2,), (5,))
        attr2 = Image1DAttr((3,), (1,), (3,))
        assert attr[:3] == attr2

    def test_squeeze(self):
        attr = Image1DAttr((1,), (2,))
        assert attr.squeeze() == Image0DAttr()
        attr = Image1DAttr((3,), (2,))
        assert attr.squeeze() == attr

    def test_locate(self):
        attr = Image1DAttr((3,), (1,))
        print(attr)
        assert attr.locate(1.5) == (1.5,)

    def test_meshgrid(self):
        attr = Image1DAttr((5,))
        assert np.array_equal(attr.meshgrid(), np.arange(5))

    def test_transpose(self):
        pass


class Test_Image2DAttr:
    def test_init(self):
        shape = (3, 2)
        center = (0, 0)
        size = (3, 2)
        dims = ('x', 'y')
        assert Image2DAttr(shape) == Image2DAttr(shape, center) == \
               Image2DAttr(shape, center, size) == Image2DAttr(shape, center, size, dims)

    def test_getitem(self):
        pass

    def test_squeeze(self):
        attr = Image2DAttr((1, 1))
        assert attr.squeeze() == Image0DAttr()
        attr = Image2DAttr((2, 1), (0, 0), (1, 2), ('x', 'y'))
        assert attr.squeeze() == Image1DAttr((2,), (0,), (1,), ('x',))
        attr = Image2DAttr((2, 3), (0, 0), (1, 2), ('x', 'y'))
        assert attr.squeeze() == attr

    def test_locate(self):
        pass

    def test_meshgrid(self):
        pass

    def test_transpose(self):
        pass


class Test_Image3DAttr:
    pass


class Test_Image4DAttr:
    pass
