#!/usr/bin/env python
# -*- encoding: utf-8 -*-

def _to_int(s):
    try:
        return int(s)
    except ValueError:
        return s


__version__ = "3.0.3"
version_info = tuple(_to_int(s) for s in __version__.split("."))
