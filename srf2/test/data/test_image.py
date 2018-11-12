import numpy as np

from srf2.data import *
from srf2.meta import *


class Test_image:
    def test_init(self):
        meta = Image_meta()
        data = np.zeros((1, 1, 1))
        assert Image() == Image(None, meta)
        assert Image() == Image(data)
        assert Image() == Image(data, meta)

    def test_property(self):
        data = np.random.random((2, 3, 4))
        meta = Image_meta(data.shape)
        image = Image(data)

        assert meta == image.meta
        assert np.array_equal(data, image.data)

    def test_transpose(self):
        data = np.random.random((2, 3, 4))
        meta = Image_meta(data.shape)
        image = Image(data, meta)
        assert np.array_equal(image.data.transpose([1, 2, 0]), image.transpose([1, 2, 0]).data)
        assert np.array_equal(image.data.transpose([1, 2, 0]),
                              image.transpose(['y', 'z', 'x']).data)

    def test_io(self):
        data = np.random.random((2, 3, 4))
        image = Image(data)
        image.save_h5()
        image2 = Image.load_h5()
        assert image == image2

    def test_slice(self):
        data = np.random.random((2, 3, 4))
        meta = Image_meta((2, 3, 4), (0, 0, 0), (2, 3, 4))
        image = Image(data, meta)
        image2 = Image(data[:, :, 1:-1], Image_meta((2, 3, 2), (0, 0, 0), (2, 3, 2)))

        assert image[:, :, 1:-1] == image2


class Test_image_2d:
    pass


class Test_image_3d:
    pass
