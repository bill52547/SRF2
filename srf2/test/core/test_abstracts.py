from srf2.core.abstracts import *


class Test_meta:
    def test_io(self):
        class Imeta(Meta):
            def __init__(self, a = (1, 2), b = ('x', 'y')):
                self.a = a
                self.b = b

        imeta = Imeta()
        imeta.save_h5()
        imeta2 = Imeta.load_h5()
        assert imeta == imeta2

        imeta = Imeta()
        imeta.save_h5('tmp_meta.h5')
        imeta2 = Imeta.load_h5('tmp_meta.h5')
        assert imeta == imeta2

        imeta = Imeta((3, 4))
        imeta.save_h5('tmp_meta.h5')
        imeta2 = Imeta.load_h5('tmp_meta.h5')
        assert imeta == imeta2


class Test_singleton:
    def test_same_id(self):
        imeta = Meta()
        imeta2 = Meta()
        assert id(imeta) != id(imeta2)

        imeta = Singleton()
        imeta2 = Singleton()
        assert id(imeta) == id(imeta2)
