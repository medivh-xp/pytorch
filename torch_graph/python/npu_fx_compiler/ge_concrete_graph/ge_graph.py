from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union, Callable
import functools
import threading
import contextlib

from torch.fx.node import Argument, Target

from .ge_ir_pb2 import GraphDef


class _GeGraphStack(threading.local):
    """A thread-local stack of objects for providing implicit defaults."""

    def __init__(self):
        super(_GeGraphStack, self).__init__()
        self.stack = []
        self._global_default_graph = None

    def get_default(self):
        """Override that returns a global default if the stack is empty."""
        if self.stack:
            return self.stack[-1]
        elif self._global_default_graph:
            return self._global_default_graph
        else:
            self._global_default_graph = GraphDef()
            return self._global_default_graph

    @contextlib.contextmanager
    def with_default(self, default):
        """A context manager for manipulating a default stack."""
        self.stack.append(default)
        try:
            yield default
        finally:
            self.stack.remove(default)


_graph_stack = _GeGraphStack()


def get_default_ge_graph():
    global _graph_stack
    return _graph_stack.get_default()


def default_ge_graph(graph):
    return _graph_stack.with_default(graph)
