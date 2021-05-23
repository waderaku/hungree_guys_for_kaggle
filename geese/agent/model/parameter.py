from typing import Tuple


class TFModuleParameter:
    pass


class BaseModelParameter(TFModuleParameter):
    def __init__(self, num_layers: int, num_filters: int, kernel_size: Tuple[int, int], bn: bool):
        self.num_layers = num_layers
        self.torusconv2d_parameter = TorusConv2dParameter(
            num_filters, kernel_size, bn)


class TorusConv2dParameter(TFModuleParameter):
    def __init__(self, num_filters: int, kernel_size: Tuple[int, int], bn: bool):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.bn = bn
