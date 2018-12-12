#!/usr/bin/env python
# encoding: utf-8
'''
@author: Minghao Guo
@license: MIT
@contact: mh.guo0111@gmail.com
@software: personal
@file: test_frozenJSON.py
@date: 12/8/2018
@desc: for personal usage
'''

import json
import os
import warnings
from urllib.request import urlopen

from srf2.core.frozenJSON import FrozenJSON

URL = 'http://www.oreilly.com/pub/sc/osconfeed'
JSON = 'osconfeed.json'


def load():
    if not os.path.exists(JSON):
        msg = 'downloading {} to {}'.format(URL, JSON)
        warnings.warn(msg)
        with urlopen(URL) as remote, open(JSON, 'wb') as local:
            local.write(remote.read())

    with open(JSON) as fp:
        return json.load(fp)


def main():
    raw_feed = load()
    feed = FrozenJSON(raw_feed)
    assert isinstance(feed, dict)
