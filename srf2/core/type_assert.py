#!/usr/bin/env python
# encoding: utf-8
'''
@author: Minghao Guo
@license: MIT
@contact: mh.guo0111@gmail.com
@software: personal
@file: type_assert.py
@date: 12/3/2018
@desc: argument type and return type assertions.
reference: PEP318: https://www.python.org/dev/peps/pep-0318/
'''

__all__ = ('arg_type_assert',)


def arg_type_assert(*types):
    def check_type(f):
        assert len(types) <= f.__code__.co_argcount, \
            "arg number %r less than asserting number %r.".format(len(types),
                                                                  f.__code__.co_argcount)

        def new_f(*args, **kwds):
            for (a, t) in zip(args, types):
                if t is None:
                    continue
                assert isinstance(a, t), "arg %r does not match %s" % (a, t)
            return f(*args, **kwds)

        new_f.__name__ = f.__name__
        return new_f

    return check_type

#
# def rtn_type_assert(rtype):
#     def check_returns(f):
#         def new_f(*args, **kwds):
#             result = f(*args, **kwds)
#             assert isinstance(result, rtype), \
#                 "return value %r does not match %s" % (result, rtype)
#             return result
#
#         new_f.func_name = f.func_name
#         return new_f
#
#     return check_returns
