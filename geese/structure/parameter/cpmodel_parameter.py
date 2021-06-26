from geese.structure.parameter import Parameter
from dataclasses import dataclass


@dataclass
class CPModelParameter(Parameter):
    dim_hidden: int
    num_layers: int
