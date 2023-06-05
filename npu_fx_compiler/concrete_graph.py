import torch
import logging
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union, Callable
from torch.fx.node import Argument, Target

class ValuePack:
    def __init__(self, meta, npu_meta=None) -> None:
        self._meta = meta
        self._npu_meta = meta if npu_meta is None else npu_meta

    @property
    def meta(self):
        return self._meta

    @property
    def npu(self):
        return self._npu_meta

    def __repr__(self) -> str:
        return f'Pack(meta:{self._meta} npu:{self._npu_meta})'


class ConcreteGraphBase(ABC):
    @abstractmethod
    def context(self):
        '''
        返回一个上下文管理器，用于在with语句中使用，表示正在构图
        '''
        pass

    @abstractmethod
    def parse_input(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        '''
        处理fx图的输入节点，其中meta_outputs为MetaTensor时的输出
        '''
        pass

    @abstractmethod
    def parse_output(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        '''
        处理fx图的输出节点，其中meta_outputs为MetaTensor时的输出，args为同时包含MetaTensor和NpuTensor输出的ValuePack
        '''
        pass

    @abstractmethod
    def parse_symlist(self, syms):
        '''
        处理fx图中的SymIntList输入，输入syms可能为int或者SymInt，或者一个ValuePack
        '''
        pass

    @abstractmethod
    def parse_node(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        '''
        处理fx图中的普通节点，其中meta_outputs为MetaTensor时的输出，对于args中的每个值：
        - 如果该入参中的值中包含symInt, 则arg为NpuTensor
        - 如果该入参为Tensor, 则arg为NpuTensor
        - 如果为基本类型，则arg为基本类型
        - 当节点的输入本身为List[Tensor]时，则为List[NpuTensor]
        '''
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        '''
        ConcreteGraphBase实例可以直接调用，用于执行fx图
        '''
        pass


def _is_symlist(arg):
    if not isinstance(arg, (list, tuple)) or len(arg) == 0:
        return False

    v = arg[0]
    if isinstance(v, (int, torch.SymInt)):
        return True
    assert isinstance(v, ValuePack)
    return isinstance(v.meta, (int, torch.SymInt))

class Logger:
    allowed_prefixes = []
    @classmethod
    def set_allowed_prefixes(cls, prefixes):
        cls.allowed_prefixes = prefixes

    def __init__(self, level=logging.DEBUG, color=True, output=sys.stdout, file=None, prefix = None):
        self.level = level
        self.color = color
        self.output = output
        self.file = file
        self.logger = None
        self.prefix = prefix

    def __enter__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(self.level)
        self.logger.disabled = Logger.allowed_prefixes and self.prefix not in Logger.allowed_prefixes

        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        if self.color:
            try:
                import colorlog
            except ImportError:
                self.color = False
            else:
                formatter = colorlog.ColoredFormatter(
                    '%(log_color)s%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    log_colors={
                        'DEBUG': 'cyan',
                        'INFO': 'green',
                        'WARNING': 'yellow',
                        'ERROR': 'red',
                        'CRITICAL': 'bold_red',
                    })

        if self.output:
            console_handler = logging.StreamHandler(self.output)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        if self.file:
            file_handler = logging.FileHandler(self.file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        self.logger.critical(f'🎵>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        self.logger.critical(f'[{self.prefix}]:')
        return self.logger

    def __exit__(self, type, value, traceback):
        self.logger.critical(f'------------------------------------------------------\n')
        if self.logger:
            for handler in self.logger.handlers:
                handler.close()
                self.logger.removeHandler(handler)