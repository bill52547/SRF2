import numpy as np

from srf2.attr.projection_attr import *


class Test_DetectorAttr:
    def test_init(self):
        param = {'shape': (1, 1), 'center': (0, 0), 'size': (1, 1), 'dims': ('u', 'v')}
        d = DetectorAttr(**param)
        d1 = DetectorAttr(param['shape'])
        d2 = DetectorAttr(param['shape'], param['center'])
        d3 = DetectorAttr(param['shape'], param['center'], param['size'])
        d4 = DetectorAttr(param['shape'], param['center'], param['size'], param['dims'])
        assert d == d1 == d2 == d3 == d4

    def test_property(self):
        param = {'shape': (1, 1), 'center': (0, 0), 'size': (1, 1), 'dims': ('u', 'v')}
        d = DetectorAttr(**param)

        assert d.shape == (1, 1)
        assert d.center == (0, 0)
        assert d.size == (1, 1)
        assert d.dims == ('u', 'v')
        assert d.unit_size == (1, 1)
        assert d.n_u == 1
        assert d.n_v == 1
        assert d.ndim == 2
        assert d.numel == 1

    def test_io(self):
        param = {'shape': (1, 1), 'center': (0, 0), 'size': (1, 1), 'dims': ('u', 'v')}
        d = DetectorAttr(**param)
        d.save_h5('tmp.h5')
        d1 = DetectorAttr.load_h5('tmp.h5')
        assert d == d1

    def test_getitem(self):
        pass

    def test_transpose(self):
        param = {'shape': (1, 2), 'center': (1, 0), 'size': (3, 1), 'dims': ('u', 'v')}
        param2 = {'shape': (2, 1), 'center': (0, 1), 'size': (1, 3), 'dims': ('v', 'u')}
        d = DetectorAttr(**param)
        d2 = DetectorAttr(**param2)
        assert d.transpose([0, 1]) == d.transpose('uv') == d.transpose(['u', 'v']) == d
        assert d.transpose() == d.transpose(['v', 'u']) == d.transpose('vu') == \
               d.transpose([1, 0]) == d2 == d.T


class Test_Detector1DAttr:
    pass


class Test_Detector2DAttr:
    pass


class Test_ProjectionAttr:
    def test_init(self):
        param = {'shape': (1, 1), 'center': (0, 0), 'size': (1, 1), 'dims': ('u', 'v')}
        d = Detector2DAttr(**param)
        p = ProjectionAttr(200, 100, 0, d)

    def test_property(self):
        param = {'shape': (1, 1), 'center': (0, 0), 'size': (1, 1), 'dims': ('u', 'v')}
        d = Detector2DAttr(**param)
        p = ProjectionAttr(200, 100, 0, d)
        assert p.detector_attr == d
        assert p.source_to_detector == 200
        assert p.source_to_image == 100
        assert p.angle == 0

    def test_source_position(self):
        param = {'shape': (1, 1), 'center': (0, 0), 'size': (1, 1), 'dims': ('u', 'v')}
        d = DetectorAttr(**param)
        p = ProjectionAttr(200, 100, 0, d)
        assert p.source_positions == (-100, 0, 0)

    def test_detector_unit_centers(self):
        pass


class Test_ProjectionFlatAttr:
    def test_detector_unit_centers(self):
        param = {'shape': (2, 1), 'center': (0, 0), 'size': (2, 1), 'dims': ('u', 'v')}
        d = Detector2DAttr(**param)
        p = ProjectionFlatAttr(200, 100, 0, d)
        assert np.array_equal(p.detector_unit_centers()[0], np.array([[100], [100]]))
        assert np.array_equal(p.detector_unit_centers()[1], np.array([[-0.5], [0.5]]))
        assert np.array_equal(p.detector_unit_centers()[2], np.array([[0], [0]]))

    def test_locate(self):
        param = {'shape': (3, 1), 'center': (0, 0), 'size': (3, 1), 'dims': ('u', 'v')}
        d = Detector2DAttr(**param)
        p = ProjectionFlatAttr(200, 100, 0, d)
        assert p.locate((0, 0, 0)) == (1, 0)
        assert p.locate((0, 0.25, 0)) == (1.5, 0)


class Test_ProjectionCurveAttr:
    def test_detector_unit_centers(self):
        pass

    def test_locate(self):
        pass
