from fx2tf_converter import TfConcreteGraph
from concrete_graph import ConcreteGraphBase, ValuePack, _is_symlist, Logger
from collections import defaultdict
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch._tensor import Tensor
from torch.utils._mode_utils import no_dispatch
from typing import Any, Dict, List, Tuple, Union
from torch.fx import Interpreter
from torch.fx.node import Argument, Target
from torch._functorch.aot_autograd import aot_module_simplified
import torch
import functools
import functools
from typing import List, Callable


def _unpack_meta_list(args):
    return [(arg.meta if (isinstance(arg, ValuePack)) else arg) for arg in args]


def _unpack_meta(args):
    unpacked = []
    for arg in args:
        if isinstance(arg, (list, tuple)) and any(isinstance(v, ValuePack) for v in arg):
            arg = _unpack_meta_list(arg)
        if isinstance(arg, ValuePack):
            unpacked.append(arg.meta)
        else:
            unpacked.append(arg)
    return list(unpacked)


def trace_print(f):
    @functools.wraps(f)
    def inner(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]):
        with Logger(prefix=f'{f.__name__}') as logger:
            logger.debug(f'target: {target}')
            for i, inp in enumerate(args):
                logger.debug(f'input {i}: {inp}')
            for k, v in kwargs.items():
                logger.debug(f'input {k}: {v}')
            result = f(self, target, args, kwargs)
            logger.debug(f'output {result}')
            return result

    return inner


class NpuGraphConverter(Interpreter):
    """
    Interpreter for collect npu graph meta from fx graph, such as sym of output, input shape ranges, etc.
    TODO: Add doc here
    """

    def __init__(self, *args, graph, **kwargs):
        super().__init__(*args, **kwargs)
        self._graph = graph

    def run(self, *args, **kwargs):
        with self._graph.context():
            super().run(*args, **kwargs)
            return self._graph

    def _wrap(self, f):
        @functools.wraps(f)
        def inner(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]):
            meta_outputs = f(target, _unpack_meta(args), kwargs)

            def _unpack_npu(args):
                unpacked = []
                for arg in args:
                    if isinstance(arg, (list, tuple)) and len(arg):
                        if _is_symlist(arg):
                            arg = self._graph.parse_symlist(arg)
                        else:
                            assert all(isinstance(v, ValuePack) for v in arg)
                            arg = [v.npu for v in arg]

                    if isinstance(arg, ValuePack):
                        unpacked.append(arg.npu)
                    else:
                        unpacked.append(arg)
                return list(unpacked)

            npu_outputs = self._graph.parse_node(target, _unpack_npu(args), kwargs, meta_outputs)
            return ValuePack(meta_outputs, npu_outputs)

        return inner

    @trace_print
    def placeholder(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        meta_input = super().placeholder(target, args=args, kwargs=kwargs)
        npu_input = self._graph.parse_input(target, args, kwargs, meta_input)
        return ValuePack(meta_input, npu_input)

    @trace_print
    def call_function(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        return self._wrap(super().call_function)(self, target, args, kwargs)

    @trace_print
    def call_method(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        return self._wrap(super().call_method)(target, args, kwargs)

    @trace_print
    def call_module(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        return self._wrap(super().call_module)(target, args, kwargs)

    @trace_print
    def output(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        meta_output = super().placeholder(target, args=_unpack_meta(args), kwargs=kwargs)
        npu_output = self._graph.parse_output(target, args, kwargs, meta_output)
        return npu_output


def npu_fx_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    with Logger(prefix='compiler inputs') as logger:
        for i, inp in enumerate(example_inputs):
            logger.debug(f'input {i}: {inp}')
        logger.warning(f'graph: {gm.graph}')
        gm.graph.print_tabular()

    concrete_graph: ConcreteGraphBase = NpuGraphConverter(gm, graph=TfConcreteGraph()).run(*example_inputs)

    with Logger(prefix="tensorflow graph") as logger:
        with open("dynamo.pbtxt", "w+") as f:
            f.write(str(concrete_graph.graph.as_graph_def()))
        logger.info(concrete_graph.graph.as_graph_def())
        logger.info(concrete_graph.outputs)

    def run_both(*args, npu_compiled_gm, original_gm, **kwargs):
        def _summary(v):
            if isinstance(v, torch.Tensor):
                return f'{type(v)}({v.size()}, {v.dtype})'
            return f'{type(v)}({v})'
        with Logger(prefix='runtime inputs') as logger:
            for i, inp in enumerate(args):
                logger.debug(f'input {i}: {_summary(inp)}')
            for k, v in kwargs.items():
                logger.debug(f'input {k}: {_summary(v)}')
            compiled_result = npu_compiled_gm(*args, **kwargs)
            original_result = original_gm(*args, **kwargs)
            logger.debug(f'npu result: {compiled_result}')
            logger.debug(f'original result: {original_result}')
            return compiled_result

    return functools.partial(run_both, npu_compiled_gm=concrete_graph, original_gm=gm)
