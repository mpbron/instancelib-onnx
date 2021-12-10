from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union
import enum
import re

def try_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    if isinstance(val, int):
        return val
    try:
        coerced = int(val)
    except (TypeError, ValueError):
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

class OnnxComponent(str, enum.Enum):
    PREPROCESSOR = "Preprocessor"
    PROBA_DECODER = "ProbaDecoder"
    PRED_DECODER = "PredDecoder"
    POST_PROCESSOR = "PostProcessor"
    CLASSIFIER = "Classifier"

class OnnxProbaDecoderTypes(str, enum.Enum):
    TENSOR = "Tensor"
    SEQMAP = "SeqMap"

class PostProcessorType(str, enum.Enum):
    IDENTITY = "Identity"
    SIGMOID = "Sigmoid"


@dataclass(eq=True, order=True, frozen=True)
class OnnxType:
    otype = OnnxTypeEnum.TYPE
    

@dataclass(eq=True, order=True, frozen=True)
class OnnxBaseType(OnnxType):
    otype = OnnxTypeEnum.BASE_TYPE
    var_type: OnnxValueType

@dataclass(eq=True, order=True, frozen=True)
class OnnxTensor(OnnxType):
    otype = OnnxTypeEnum.TENSOR
    dtype: OnnxDType

@dataclass(eq=True, order=True, frozen=True)
class OnnxSequence(OnnxType):
    otype = OnnxTypeEnum.SEQUENCE
    item_type: OnnxType

@dataclass(eq=True, order=True, frozen=True)
class OnnxMap(OnnxType):
    otype = OnnxTypeEnum.MAP
    key_type: OnnxType
    value_type: OnnxType

Shape = Sequence[Optional[int]]

@dataclass(eq=True, order=True, frozen=True)
class OnnxVariable:
    name: str
    vartype: OnnxType
    shape: Optional[Shape]



