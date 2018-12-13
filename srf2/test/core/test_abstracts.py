import numpy as np
from numba import cuda

from srf2.core.abstracts import *


def ensure_gpu_env(f):
    if cuda.gpus is None:
        return
    f


class Attr(AttributeWithShape):
    def __init__(self, str1=None, str2=None, num1=None, num2=None):
        if not str1:
            self.str1 = 'abc'
            self.str2 = ('x', 'y', 'z')
            self.num1 = 123456
            self.num2 = (1, 2, 3)
        else:
            self.str1 = str1
            self.str2 = str2
            self.num1 = num1
            self.num2 = num2


class Test_Attribute:
    def test_init(self):
        obj = Attr()
        assert obj.str1 == 'abc'
        assert obj.str2 == ('x', 'y', 'z')
        assert obj.num1 == 123456
        assert obj.num2 == (1, 2, 3)

    def test_eq(self):
        obj1 = Attr()
        obj2 = Attr('abc', ('x', 'y', 'z'), 123456, (1, 2, 3))
        assert obj1 == obj2

    def test_io(self):
        path = 'tmp.h5'
        obj = Attr()
        obj.save_h5(path)
        obj2 = Attr.load_h5(path)
        assert obj == obj2
        obj.save_h5()
        obj2 = Attr.load_h5()
        assert obj == obj2

    def test_print(self):
        obj = Attr()
        print(obj)


class Attr2(AttributeWithShape):
    def __init__(self, shape=(5,)):
        self._shape = shape

    @property
    def shape(self):
        return self._shape


class Obj1(ObjectWithAttrData):
    _attr = Attr2()

    def __init__(self, attr=Attr2(), data=np.zeros(5, )):
        super().__init__(attr, data)

    def map(self, f):
        return self.__class__(*f(self.attr, self.data))


class Test_Object:
    def test_init(self):
        attr = Attr2()
        o1 = Obj1()
        o2 = Obj1(attr, np.array([5] * 5))

    def test_eq(self):
        attr = Attr2()
        o1 = Obj1(attr)
        o2 = Obj1(o1.attr, np.zeros(5, ))
        assert o1 == o2

    def test_io(self):
        attr = Attr2()
        o1 = Obj1(attr)
        o1.save_h5('tmp.h5')
        o2 = Obj1.load_h5('tmp.h5')
        assert o1 == o2

    def test_print(self):
        obj = Obj1()
        # print(obj)

    def test_neg(self):
        attr = Attr2()

        o1 = Obj1(attr, np.ones(5, ))
        o2 = Obj1(attr, -np.ones(5, ))
        assert -o1 == o2

    def test_pos(self):
        attr = Attr2()

        o1 = Obj1(attr, np.ones(5, ))
        o2 = Obj1(attr, np.ones(5, ))
        assert o1 == o2

    def test_add(self):
        attr = Attr2()

        o1 = Obj1(attr, np.ones(5, ))
        o2 = Obj1(attr, np.ones(5, ) * 2)
        assert o1 + o2 == Obj1(attr, np.ones(5, ) * 3)
        assert o2 + o1 == Obj1(attr, np.ones(5, ) * 3)
        assert o1 + 1 == o2
        assert 1 + o1 == o2

    def test_sub(self):
        attr = Attr2()

        o1 = Obj1(attr, np.ones(5, ))
        o2 = Obj1(attr, np.ones(5, ) * 2)
        assert o2 - o1 == o1
        assert o2 - 1 == o1

    def test_mul(self):
        attr = Attr2()

        o1 = Obj1(attr, np.ones(5, ))
        o2 = Obj1(attr, np.ones(5, ) * 2)
        assert o1 * o2 == o2
        assert o2 * o1 == o2
        assert o1 * 2 == o2
        assert 2 * o1 == o2

    def test_div(self):
        attr = Attr2()

        o1 = Obj1(attr, np.ones(5, ))
        o2 = Obj1(attr, np.ones(5, ) * 2)
        assert o2 / o1 == o2
        assert o2 / 2 == o1

    def test_iadd(self):
        attr = Attr2()

        o1 = Obj1(attr, np.ones(5, ))
        o2 = Obj1(attr, np.ones(5, ) * 2)
        o1 += o2
        o2 += 1
        assert o1 == Obj1(attr, np.ones(5, ) * 3)
        assert o2 == Obj1(attr, np.ones(5, ) * 3)

    def test_isub(self):
        attr = Attr2()

        o1 = Obj1(attr, np.ones(5, ))
        o2 = Obj1(attr, np.ones(5, ) * 2)
        o1 -= o2
        o2 -= 1
        assert o1 == Obj1(attr, np.ones(5, ) * (-1))
        assert o2 == Obj1(attr, np.ones(5, ) * 1)

    def test_imul(self):
        attr = Attr2()

        o1 = Obj1(attr, np.ones(5, ))
        o2 = Obj1(attr, np.ones(5, ) * 2)
        o1 *= o2
        o2 *= 2
        assert o1 == Obj1(attr, np.ones(5, ) * 2)
        assert o2 == Obj1(attr, np.ones(5, ) * 4)

    def test_idiv(self):
        attr = Attr2()

        o1 = Obj1(attr, np.ones(5, ))
        o2 = Obj1(attr, np.ones(5, ) * 2)
        o1 /= o2
        o2 /= 2
        assert o1 == Obj1(attr, np.ones(5, ) * 0.5)
        assert o2 == Obj1(attr, np.ones(5, ) * 1)

    @ensure_gpu_env
    def test_neg_cuda(self):
        attr = Attr2()
        o1 = -Obj1(attr, np.ones(5, )).to_device()
        attr = Attr2()
        o2 = Obj1(attr, -np.ones(5, ))
        assert o1.to_host() == o2

    @ensure_gpu_env
    def test_pos_cuda(self):
        attr = Attr2()
        o1 = Obj1(attr, np.ones(5, )).to_device()
        attr = Attr2()
        o2 = Obj1(attr, np.ones(5, )).to_device()
        assert o1.to_host() == o2.to_host()

    @ensure_gpu_env
    def test_add_cuda(self):
        attr1 = Attr2()
        o1 = Obj1(attr1, np.ones(5, )).to_device()
        attr2 = Attr2()
        o2 = Obj1(attr2, np.ones(5, ) * 2).to_device()
        attr3 = Attr2()
        o3 = Obj1(attr3, np.ones(5, ) * 3)

        assert (o1 + o2).to_host() == o3
        assert (o1 + o2.to_host()).to_host() == o3
        assert (o1 + 2).to_host() == o3

    @ensure_gpu_env
    def test_sub_cuda(self):
        attr1 = Attr2()
        o1 = Obj1(attr1, np.ones(5, )).to_device()
        attr2 = Attr2()
        o2 = Obj1(attr2, np.ones(5, ) * 2).to_device()
        attr3 = Attr2()
        o3 = Obj1(attr3, np.ones(5, ) * -1)

        assert (o1 - o2).to_host() == o3
        assert (o1 - o2.to_host()).to_host() == o3
        assert (o1 - 2).to_host() == o3

    @ensure_gpu_env
    def test_mul_cuda(self):
        attr1 = Attr2()
        o1 = Obj1(attr1, np.ones(5, ) * 2).to_device()
        attr2 = Attr2()
        o2 = Obj1(attr2, np.ones(5, ) * 3).to_device()
        attr3 = Attr2()
        o3 = Obj1(attr3, np.ones(5, ) * 6)

        assert (o1 * o2).to_host() == o3
        assert (o1 * o2.to_host()).to_host() == o3
        assert (o1 * 3).to_host() == o3

    @ensure_gpu_env
    def test_div_cuda(self):
        attr1 = Attr2()
        o1 = Obj1(attr1, np.ones(5, )).to_device()
        attr2 = Attr2()
        o2 = Obj1(attr2, np.ones(5, ) * 2).to_device()
        attr3 = Attr2()
        o3 = Obj1(attr3, np.ones(5, ) * 0.5)

        assert (o1 / o2).to_host() == o3
        assert (o1 / o2.to_host()).to_host() == o3
        assert (o1 / 2).to_host() == o3

    @ensure_gpu_env
    def test_iadd_cuda(self):
        attr1 = Attr2()
        o1 = Obj1(attr1, np.ones(5, )).to_device()
        attr2 = Attr2()
        o2 = Obj1(attr2, np.ones(5, ) * 2).to_device()
        attr3 = Attr2()
        o3 = Obj1(attr3, np.ones(5, ) * 3)
        o1 += o2
        assert o1.to_host() == o3

        attr1 = Attr2()
        o1 = Obj1(attr1, np.ones(5, )).to_device()
        attr3 = Attr2()
        o3 = Obj1(attr3, np.ones(5, ) * 3)
        o1 += 2
        assert o1.to_host() == o3

        attr1 = Attr2()
        o1 = Obj1(attr1, np.ones(5, )).to_device()
        attr2 = Attr2()
        o2 = Obj1(attr2, np.ones(5, ) * 2).to_device()
        attr3 = Attr2()
        o3 = Obj1(attr3, np.ones(5, ) * 3)
        o1 += o2.to_host()
        assert o1.to_host() == o3

    @ensure_gpu_env
    def test_isub_cuda(self):
        attr1 = Attr2()
        o1 = Obj1(attr1, np.ones(5, )).to_device()
        attr2 = Attr2()
        o2 = Obj1(attr2, np.ones(5, ) * 2).to_device()
        attr3 = Attr2()
        o3 = Obj1(attr3, np.ones(5, ) * -1)
        o1 -= o2
        assert o1.to_host() == o3

        attr1 = Attr2()
        o1 = Obj1(attr1, np.ones(5, )).to_device()
        attr3 = Attr2()
        o3 = Obj1(attr3, np.ones(5, ) * -1)
        o1 -= 2
        assert o1.to_host() == o3

        attr1 = Attr2()
        o1 = Obj1(attr1, np.ones(5, )).to_device()
        attr2 = Attr2()
        o2 = Obj1(attr2, np.ones(5, ) * 2).to_device()
        attr3 = Attr2()
        o3 = Obj1(attr3, np.ones(5, ) * -1)
        o1 -= o2.to_host()
        assert o1.to_host() == o3

    @ensure_gpu_env
    def test_imul_cuda(self):
        attr1 = Attr2()
        o1 = Obj1(attr1, np.ones(5, ) * 2).to_device()
        attr2 = Attr2()
        o2 = Obj1(attr2, np.ones(5, ) * 3).to_device()
        attr3 = Attr2()
        o3 = Obj1(attr3, np.ones(5, ) * 6)
        o1 *= o2
        assert o1.to_host() == o3

        attr1 = Attr2()
        o1 = Obj1(attr1, np.ones(5, ) * 2).to_device()
        attr3 = Attr2()
        o3 = Obj1(attr3, np.ones(5, ) * 6)
        o1 *= 3
        assert o1.to_host() == o3

        attr1 = Attr2()
        o1 = Obj1(attr1, np.ones(5, ) * 2).to_device()
        attr2 = Attr2()
        o2 = Obj1(attr2, np.ones(5, ) * 3).to_device()
        attr3 = Attr2()
        o3 = Obj1(attr3, np.ones(5, ) * 6)
        o1 *= o2.to_host()
        assert o1.to_host() == o3

    @ensure_gpu_env
    def test_idiv_cuda(self):
        attr1 = Attr2()
        o1 = Obj1(attr1, np.ones(5, ) * 2).to_device()
        attr2 = Attr2()
        o2 = Obj1(attr2, np.ones(5, ) * 4).to_device()
        attr3 = Attr2()
        o3 = Obj1(attr3, np.ones(5, ) * 0.5)
        o1 /= o2
        assert o1.to_host() == o3

        attr1 = Attr2()
        o1 = Obj1(attr1, np.ones(5, ) * 2).to_device()
        attr3 = Attr2()
        o3 = Obj1(attr3, np.ones(5, ) * 0.5)
        o1 /= 4
        assert o1.to_host() == o3

        attr1 = Attr2()
        o1 = Obj1(attr1, np.ones(5, ) * 2).to_device()
        attr2 = Attr2()
        o2 = Obj1(attr2, np.ones(5, ) * 4).to_device()
        attr3 = Attr2()
        o3 = Obj1(attr3, np.ones(5, ) * 0.5)
        o1 /= o2.to_host()
        assert o1.to_host() == o3
