#!/usr/bin/env python
# encoding: utf-8
'''
@author: Minghao Guo
@license: MIT
@contact: mh.guo0111@gmail.com
@software: personal
@file: frozenJSON.py
@date: 12/8/2018
@desc: An read-only interface, to get json object in manners of an object
<Fluent Python> Example 19-5
'''

from collections import abc
from keyword import iskeyword


class FrozenJSON:
    def __new__(cls, arg):
        if isinstance(arg, abc.Mapping):
            return super().__new__(cls)
        elif isinstance(arg, abc.MutableSequence):
            return [cls(item) for item in arg]
        else:
            return arg

    def __init__(self, mapping):
        self.__data = {}
        for key, value in mapping.items():
            if iskeyword(key):
                key += '_'
            self.__data[key] = value

    def __getattr__(self, item):
        if hasattr(self.__data, item):
            return getattr(self.__data, item)
        elif item in self.__data.keys():
            return FrozenJSON(self.__data[item])
        else:
            raise AttributeError

# class FrozenJSONRelease:
#     def __init__(self, mapping):
#         self.__data = dict(mapping)
#
#     def __getattr__(self, item):
#         if hasattr(self.__data, item):
#             return getattr(self.__data, item)
#         else:
#             return FrozenJSON.build(self.__data[item])
#
#     @classmethod
#     def build(cls, obj):
#         if isinstance(obj, abc.Mapping):
#             return cls(obj)
#         elif isinstance(obj, abc.MutableSequence):
#             return [cls.build(item) for item in obj]
#         else:
#             return obj

# TODO expand a JSON file to a dictionary while loading
