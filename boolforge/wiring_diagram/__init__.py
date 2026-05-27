#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wiring-diagram representation for Boolean networks.

This module defines the :class:`~boolforge.WiringDiagram` class, which encodes
the directed regulatory topology of a Boolean network independently of any
Boolean update functions.

A wiring diagram specifies, for each node, the set of regulating nodes
(predecessors). Nodes with no regulators are source nodes in the wiring diagram.
Whether such nodes act as constants or identity nodes in Boolean networks
can only be determined after Boolean update functions are assigned.
"""

from .core import WiringDiagram

__all__ = [
    "WiringDiagram",
]
