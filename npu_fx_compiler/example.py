
from concrete_graph import Logger
from typing import Any, Dict, List, Tuple, Union
from torch._functorch.aot_autograd import aot_module_simplified
import torch
import functools

from npu_fx_compiler import npu_fx_compiler


def npu_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    with Logger(prefix='backend inputs') as logger:
        for i, inp in enumerate(example_inputs):
            logger.debug(f'input {i}: {inp}')
    return aot_module_simplified(gm, example_inputs, fw_compiler=npu_fx_compiler)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = torch.cat([torch.ones(x.size()), torch.ones(y.size())])
        x = torch.ones(x.size())
        x = torch.split(x, 2)
        return x[-1], x[0]


model = Model()
model = torch.compile(model, backend=npu_backend, dynamic=True)
model(torch.randn(2, 2), torch.randn(2, 2))
model(torch.randn(3, 3), torch.randn(3, 3))
