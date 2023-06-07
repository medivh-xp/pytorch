#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions used for NPU device"""

import os
import atexit
import threading
import logging

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _npu_fx_compiler
else:
    import _npu_fx_compiler


def stupid_repeat(word, times):
    """Simple repeated"""
    return _npu_fx_compiler.StupidRepeat(word, times)

def parse_serialized_ge_graph(serialized_ge_graph):
    """Simple repeated"""
    return _npu_fx_compiler.ParseSerializedGraph(serialized_ge_graph)

