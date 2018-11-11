from srf2.meta.projection_meta import *


class Test_projection_meta:
    def test_init(self):
        pmeta = Projection_meta((1, 1, 1), (0, 0, 0), (1, 1, 1), 99.0, 119.0, 33.4, 1,
                                16, 0.0)
        assert pmeta == Projection_meta()

    def test_property(self):
        pmeta = Projection_meta()
        assert pmeta.shape == (1, 1, 1)
        assert pmeta.center == (0, 0, 0)
        assert pmeta.size == (1, 1, 1)
        assert pmeta.unit_size == (1, 1, 1)
        assert pmeta.inner_radius == 99.0
        assert pmeta.outer_radius == 119.0
        assert pmeta.axial_length == 33.4
        assert pmeta.nb_rings == 1
        assert pmeta.nb_blocks_per_ring == 16
        assert pmeta.gap == 0.0
        assert pmeta.n_sensors_per_block == 1
        assert pmeta.n_sensors_per_ring == 16
        assert pmeta.n_sensors_all == 16

    def test_io(self):
        pmeta = Projection_meta()
        pmeta.save_h5()
        pmeta2 = Projection_meta.load_h5()
        assert pmeta == pmeta2


class Test_sinogram_projection_meta:
    def test_transfer(self):
        sino_meta = Sinogram_projection_meta()
        list_meta2 = sino_meta.transfer('listmode')
        list_meta = Listmode_projection_meta()
        assert list_meta == list_meta2

        sino_meta2 = sino_meta.transfer('sinogram')
        assert sino_meta == sino_meta2


class Test_listmode_projection_meta:
    def test_transfer(self):
        list_meta = Listmode_projection_meta()
        sino_meta2 = list_meta.transfer('sinogram')
        sino_meta = Sinogram_projection_meta()
        assert sino_meta == sino_meta2

        list_meta2 = list_meta.transfer('listmode')
        assert list_meta == list_meta2
