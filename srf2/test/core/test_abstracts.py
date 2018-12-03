import numpy as np

from srf2.core.abstracts import *


class Attr(Attribute):
    def __init__(self, str1 = None, str2 = None, num1 = None, num2 = None):
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


class Obj1(ObjectWithAttrData):
    _attr = Attr()

    def __init__(self, _data = np.zeros(1, ), _attr = Attr()):
        self._data = _data
        self._attr = _attr

    def map(self, f):
        return self.__class__(*f(self._data, self._attr))


class Test_Object:
    def test_init(self):
        o1 = Obj1()
        o2 = Obj1(np.array([5]))

    def test_eq(self):
        o1 = Obj1()
        o2 = Obj1(np.zeros(1, ))
        assert o1 == o2

    def test_io(self):
        o1 = Obj1()
        o1.save_h5('tmp.h5')
        o2 = Obj1.load_h5('tmp.h5')
        assert o1 == o2

    def test_print(self):
        obj = Obj1()
        # print(obj)

    def test_neg(self):
        o1 = Obj1(np.ones(5, ))
        o2 = Obj1(-np.ones(5, ))
        assert -o1 == o2

    def test_pos(self):
        o1 = Obj1(np.ones(5, ))
        o2 = Obj1(np.ones(5, ))
        assert o1 == o2

    def test_add(self):
        o1 = Obj1(np.ones(5, ))
        o2 = Obj1(np.ones(5, ) * 2)
        assert o1 + o2 == Obj1(np.ones(5, ) * 3)
        assert o2 + o1 == Obj1(np.ones(5, ) * 3)
        assert o1 + 1 == o2
        assert 1 + o1 == o2

    def test_sub(self):
        o1 = Obj1(np.ones(5, ))
        o2 = Obj1(np.ones(5, ) * 2)
        assert o2 - o1 == o1
        assert o2 - 1 == o1

    def test_mul(self):
        o1 = Obj1(np.ones(5, ))
        o2 = Obj1(np.ones(5, ) * 2)
        assert o1 * o2 == o2
        assert o2 * o1 == o2
        assert o1 * 2 == o2
        assert 2 * o1 == o2

    def test_div(self):
        o1 = Obj1(np.ones(5, ))
        o2 = Obj1(np.ones(5, ) * 2)
        assert o2 / o1 == o2
        assert o2 / 2 == o1
