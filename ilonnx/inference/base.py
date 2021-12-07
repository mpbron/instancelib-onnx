from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union
import enum
import re

def try_int(val: Any) -> Optional[int]:
    if isinstance(val, int):
        return val
    try:
        coerced = int(val)
    except ValueError:
        return None
    else:
        return coerced

def get_shape(vinfo: Any) -> Sequence[Optional[int]]:
    return list(map(try_int, vinfo.shape))

def get_dimensions(vinfo: Any) -> int:
    return len(get_shape(vinfo))

class OnnxValueType(str, enum.Enum):
    INT64 = "int64"
    DOUBLE = "double"
    FLOAT = "float"
    STRING = "string"

class OnnxTypeEnum(str, enum.Enum):
    TYPE = "Type"
    BASE_TYPE = "BaseType"
    TENSOR = "Tensor"
    SEQUENCE = "Sequence"
    MAP = "Map"

class OnnxDType(str, enum.Enum):
    INT64 = "int64"
    FLOAT = "float"
    DOUBLE = "double"
    STRING = "string"


    @classmethod
    def get_dtype(cls, vinfo: Any) -> OnnxValueType:
        vtype: str = vinfo.type
        subtype: str = re.search(r"\(([A-Za-z0-9_]+)\)", vtype).group(1)
        return OnnxDType(subtype)

class OnnxComponent(str, enum.Enum):
    INPUT_TRANSLATOR = "InputTranslator"
    OUTPUT_TRANSLATOR = "OutputTranslator"
    POSTPROCESSOR = "PostProcessor"
    ENCODER = "Encoder"
    VARIABLE = "Variable"
    TENSOR_VARIABLE = "TensorVariable"
    SEQMAP_VARIABLE = "SeqMapVariable"


@dataclass
class OnnxType:
    otype = OnnxTypeEnum.TYPE
    

@dataclass
class OnnxBaseType(OnnxType):
    otype = OnnxTypeEnum.BASE_TYPE
    var_type: OnnxValueType

@dataclass
class OnnxTensor(OnnxType):
    otype = OnnxTypeEnum.TENSOR
    dtype: OnnxDType

@dataclass
class OnnxSequence(OnnxType):
    otype = OnnxTypeEnum.SEQUENCE
    item_type: OnnxType

@dataclass
class OnnxMap(OnnxType):
    otype = OnnxTypeEnum.MAP
    key_type: OnnxType
    value_type: OnnxType



