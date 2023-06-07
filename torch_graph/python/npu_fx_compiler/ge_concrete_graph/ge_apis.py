from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union, Callable
import functools

from .ge_ir_pb2 import GraphDef, OpDef, DataType, TensorDescriptor, TensorDef
from .ge_graph import get_default_ge_graph

_g_name_dict = dict()


def _next_unique_name(name: str, op: str):
    if name is not None:
        return name

    if op not in _g_name_dict:
        _g_name_dict[op] = 0
        return op
    _g_name_dict[op] += 1
    return f'{op}_{_g_name_dict[op]}'


class Tensor:
    def __init__(self, node: OpDef, index: int = 0):
        self._tensor = f'{node.name}:{index}'

    @property
    def tensor(self):
        return self._tensor

    def __repr__(self) -> str:
        return f'Tensor({self.tensor})'


def auto_convert_to_tensor(num_tensor_inputs):
    def inner(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            args = list(args)
            for i, arg in enumerate(args):
                if i < num_tensor_inputs and not isinstance(arg, Tensor):
                    raise RuntimeError(
                        f"{func.__name__} expect input {i} to be Tensor, but got {type(arg)}")
            return func(*args, **kwargs)
        return wrapper
    return inner


@auto_convert_to_tensor(0)
def placeholder(name: str, index: int, dtype: 'DataType') -> Tensor:
    op = get_default_ge_graph().op.add()
    op.type = "Data"
    op.name = _next_unique_name(name, "Data")
    op.attr["index"].i = index
    return Tensor(op)


@auto_convert_to_tensor(1)
def shape(input: Tensor, name=None) -> Tensor:
    op = get_default_ge_graph().op.add()
    op.type = "Shape"
    op.name = _next_unique_name(name, "Shape")
    op.input.append(input.tensor)
    return Tensor(op)


@auto_convert_to_tensor(3)
def gather(input: Tensor, indices: Tensor, axis: Tensor, name=None) -> Tensor:
    op = get_default_ge_graph().op.add()
    op.type = "Gather"
    op.name = _next_unique_name(name, "Gather")
    op.input.append(input.tensor)
    op.input.append(indices.tensor)
    op.input.append(axis.tensor)
    return Tensor(op)


@auto_convert_to_tensor(2)
def fill(dims: Tensor, value: Tensor, name=None) -> Tensor:
    op = get_default_ge_graph().op.add()
    op.type = "Fill"
    op.name = _next_unique_name(name, "Fill")
    op.input.append(dims.tensor)
    op.input.append(value.tensor)
    return Tensor(op)


@auto_convert_to_tensor(2)
def add(x: Tensor, y: Tensor, name=None) -> Tensor:
    op = get_default_ge_graph().op.add()
    op.type = "Add"
    op.name = _next_unique_name(name, "Add")
    op.input.append(x.tensor)
    op.input.append(y.tensor)
    return Tensor(op)


@auto_convert_to_tensor(0)
def constant(v: Any, name=None) -> Tensor:
    op = get_default_ge_graph().op.add()
    op.type = "Constant"
    op.name = _next_unique_name(name, "Constant")
    op.attr["value"].t = TensorDef()
    return Tensor(op)
