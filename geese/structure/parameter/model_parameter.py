from __future__ import annotations
from typing import Tuple
from geese.structure.parameter.parameter import Parameter


class TFModuleParameter(Parameter):
    pass


class BaseModelParameter(TFModuleParameter):
    def __init__(
        self,
        num_layers: int,
        num_filters: int,
        kernel_size: Tuple[int, int],
        bn: bool,
        use_gpu: bool,
    ):
        self._num_layers = num_layers
        self._torusconv2d_parameter = TorusConv2dParameter(
            num_filters, kernel_size, bn, use_gpu)

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def torusconv2d_parameter(self) -> TorusConv2dParameter:
        return self._torusconv2d_parameter


class TorusConv2dParameter(TFModuleParameter):
    def __init__(self, num_filters: int, kernel_size: Tuple[int, int], bn: bool, use_gpu: bool):
        self._num_filters = num_filters
        self._kernel_size = kernel_size
        self._bn = bn
        self._use_gpu = use_gpu

    @property
    def num_filters(self) -> int:
        return self._num_filters

    @property
    def kernel_size(self) -> Tuple[int, int]:
        return self._kernel_size

    @property
    def bn(self) -> bool:
        return self._bn

    @property
    def use_gpu(self) -> bool:
        return self._use_gpu
