import numpy as np

from srf2.attr.image_attr import *
from srf2.data.image import *


class Test_image:
    def test_init(self):
        attr = ImageAttr((5,))
        assert not Image(attr)
        data = np.zeros((5,))
        Image(attr, data)

    def test_property(self):
        pass

    def test_bool(self):
        pass

    def test_map(self):
        pass

    def test_eq(self):
        pass

    def test_transpose(self):
        attr = ImageAttr((5, 5)).squeeze()
        img = Image(attr)
        assert img.transpose() == img.transpose([1, 0]) == img.transpose('yx') == \
               img.transpose(['y', 'x']) == Image(attr.transpose()) == img.T
        assert img.transpose([0, 1]) == img.transpose('xy') == img.transpose(['x', 'y']) == img
        data = np.random.random(attr.shape)
        img = Image(attr, data)
        assert img.transpose() == img.transpose([1, 0]) == img.transpose('yx') == \
               img.transpose(['y', 'x']) == Image(attr.transpose(), data.transpose()) == img.T
        assert img.transpose([0, 1]) == img.transpose('xy') == img.transpose(['x', 'y']) == img

    def test_squeeze(self):
        attr = ImageAttr((5, 1)).squeeze()
        data = np.random.random(attr.shape)

        attr1 = ImageAttr((5,)).squeeze()
        data1 = data

        assert Image(attr, data).squeeze() == Image(attr1, data1).squeeze()
        assert isinstance(Image(attr, data).squeeze(), Image1D)


class Test_Image0D:
    pass


class Test_Image1D:
    pass


class Test_Image2D:
    pass


class Test_Image3D:
    pass
#       ass
#     def test_property(self):
#         data = np.random.random((2, 3, 4))
#         meta = Image_meta(data.shape)
#         image = Image(data)
#
#         assert meta == image.meta
#         assert np.array_equal(data, image.data)
#
#     def test_io(self):
#         data = np.random.random((2, 3, 4))
#         image = Image(data)
#         image.save_h5()
#         image2 = Image.load_h5()
#         assert image == image2
#
#     def test_slice(self):
#         data = np.random.random((2, 3, 4))
#         meta = Image_meta((2, 3, 4), (0, 0, 0), (2, 3, 4))
#         image = Image(data, meta)
#         image2 = Image(data[:, :, 1:-1], Image_meta((2, 3, 2), (0, 0, 0), (2, 3, 2)))
#
#         assert image[:, :, 1:-1] == image2
#
#
# class Test_image_2d:
#     pass
#
#
# class Test_image_3d:
#     def test_transpose(self):
#         data = np.random.random((2, 3, 4))
#         meta = Image_meta_3d(data.shape)
#         image = Image_3d(data, meta)
#         assert np.array_equal(image.data.transpose([1, 2, 0]), image.transpose([1, 2, 0]).data)
#         assert np.array_equal(image.data.transpose([1, 2, 0]),
#                               image.transpose(['y', 'z', 'x']).data)
