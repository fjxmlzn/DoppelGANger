from enum import Enum


class OutputType(Enum):
    CONTINUOUS = "CONTINUOUS"
    DISCRETE = "DISCRETE"


class Normalization(Enum):
    ZERO_ONE = "ZERO_ONE"
    MINUSONE_ONE = "MINUSONE_ONE"


class Output(object):
    def __init__(self, type_, dim, normalization=None, is_gen_flag=False):
        self.type_ = type_
        self.dim = dim
        self.normalization = normalization
        self.is_gen_flag = is_gen_flag

        if type_ == OutputType.CONTINUOUS and normalization is None:
            raise Exception("normalization must be set for continuous output")
