from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union, Callable
import functools
import threading
import contextlib

import torch
from torch.fx.node import Argument, Target

from npu_fx_compiler.core.concrete_graph import ConcreteGraphBase, ValuePack, _is_symlist, Logger
from npu_fx_compiler.core.backend import parse_serialized_ge_graph
from .ge_ir_pb2 import GraphDef, DataType, TensorDescriptor, TensorDef
from .ge_graph import default_ge_graph
from . import ge_apis as ge

__CONVERTERS = defaultdict(None)


def _get_converter(name: Callable):
    global __CONVERTERS
    return __CONVERTERS[name]


def register_fx_node_ge_converter(aten_op, converter: Callable = None):
    if converter is not None:
        global __CONVERTERS
        __CONVERTERS.update({aten_op: converter})
        return converter

    def register_demo(f, key):
        global __CONVERTERS
        __CONVERTERS.update({key: f})
        return f

    return functools.partial(register_demo, key=aten_op)


def _torch_type_to_ge_type(dtype):
    if dtype == torch.float32:
        return DataType.DT_FLOAT
    elif dtype == torch.int32:
        return DataType.DT_INT32
    elif dtype == torch.bool:
        return DataType.DT_BOOL
    else:
        raise RuntimeError(f"Unsupported torch type {dtype} by tf")


class GeConcreteGraph(ConcreteGraphBase):
    def __init__(self, graph=None):
        self._graph = GraphDef() if graph is None else graph
        self._fx_outputs = []
        self._fx_outputs_mapping = dict()
        self._outputs = []
        self._inputs = []

    def context(self):
        return default_ge_graph(self.graph)

    def parse_input(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        if isinstance(meta_outputs, torch.SymInt):
            self._inputs.append(ge.placeholder(
                name=target, index=len(self._inputs), dtype=DataType.DT_INT32))
        else:
            assert isinstance(meta_outputs, torch.Tensor)
            self._inputs.append(ge.placeholder(
                name=target, index=len(self._inputs), dtype=_torch_type_to_ge_type(meta_outputs.dtype)))
        return self._inputs[-1]

    def parse_output(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        assert isinstance(args, (list, tuple)) and len(args) == 1
        args = args[0]
        for arg in args:
            arg = arg.npu if isinstance(arg, ValuePack) else arg
            if isinstance(arg, ge.Tensor):
                self._fx_outputs_mapping[len(
                    self._fx_outputs)] = len(self._outputs)
                self._outputs.append(arg)
            self._fx_outputs.append(arg)

        return args

    def parse_symlist(self, syms):
        concat_inputs = []
        for sym in syms:
            if isinstance(sym, ValuePack):
                concat_inputs.append(sym.npu)
            else:
                assert isinstance(sym, int)
                concat_inputs.append(ge.constant(sym, name=f"{sym}"))

        return ge.stack(concat_inputs, axis=0)

    def parse_node(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        converter = _get_converter(target)
        if converter is None:
            raise RuntimeError(f"Unsupported torch op {target} by ge")
        return converter(args, kwargs, meta_outputs)

    def dump(self, path: str):
        with Logger(prefix="ge graph") as logger:
            with open(path, "w+") as f:
                f.write(str(self.graph))
            logger.info(self.graph)
            logger.info(self.outputs)

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def graph(self):
        return self._graph

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        print("-------------------------")
        parse_serialized_ge_graph(self.graph.SerializeToString())
        print("-------------------------")
        for k, v in self._fx_outputs_mapping.items():
            self._fx_outputs[k] = torch.Tensor(1)
        return tuple(self._fx_outputs)

        # with tf.Session(graph=self.graph) as sess:
        #     args = [arg.numpy() if isinstance(arg, torch.Tensor)
        #             else arg for arg in args]
        #     tf_outputs = sess.run(
        #         self.outputs, feed_dict=dict(zip(self.inputs, args)))
        #     for k, v in self._fx_outputs_mapping.items():
        #         self._fx_outputs[k] = torch.Tensor(tf_outputs[v])
        #     return tuple(self._fx_outputs)


@register_fx_node_ge_converter(torch.ops.aten.sym_size)
def conveter_sym_size(args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
    shape = ge.shape(args[0])
    return ge.gather(shape, args[1])


@register_fx_node_ge_converter(torch.ops.aten.cat.default)
def conveter_cat(args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
    return ge.concat(args[0], 0 if len(args) == 1 else args[1])


@register_fx_node_ge_converter(torch.ops.aten.ones.default)
def conveter_ones(args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
    return ge.fill(args[0], ge.constant(1))

@register_fx_node_ge_converter(torch.ops.aten.add.Tensor)
def conveter_add(args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
    return ge.add(args[0], args[1])

@register_fx_node_ge_converter(torch.ops.aten.split.Tensor)
def conveter_split(args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
    split_sizes = args[1]
    if isinstance(split_sizes, int):
        split_sizes = [args[1] for _ in range(len(meta_outputs))]
        split_sizes[-1] = -1
    return ge.split(args[0], split_sizes)
