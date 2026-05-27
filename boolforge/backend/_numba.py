#!/usr/bin/env python3
# -*- coding: utf-8 -*-

try:
    import numba
    from numba import njit
    from numba.typed import List
    int64 = numba.int64
    __LOADED_NUMBA__ = True
except ModuleNotFoundError:
    __LOADED_NUMBA__ = False