from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from geese.structure.parameter.parameter import Parameter


class TFModuleParameter(Parameter):
    pass


@dataclass(frozen=True)
class BaseModelParameter(TFModuleParameter):
    num_layers: int
    num_filters: int
    kernel_size: Tuple[int, int]
    bn: bool
    use_gpu: bool

    @property
    def torusconv2d_parameter(self) -> TorusConv2dParameter:
        return TorusConv2dParameter(
            self.num_filters, self.kernel_size, self.bn, self.use_gpu)


@dataclass(frozen=True)
class TorusConv2dParameter(TFModuleParameter):
    num_filters: int
    kernel_size: Tuple[int, int]
    bn: bool
    use_gpu: bool
