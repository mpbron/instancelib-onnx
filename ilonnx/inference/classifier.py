from abc import ABC
from ctypes import Union
from typing import Any, Generic, Iterable, TypeVar
from instancelib.machinelearning.base import AbstractClassifier
from instancelib.typehints.typevars import IT, KT, DT, VT, RT, LT, LMT, PMT
from instancelib import InstanceProvider, Instance


IT = TypeVar("IT", 
             bound="Instance[Any,Any,Any,Any]", covariant = True)

InstanceInput = Union[InstanceProvider[IT, KT, DT, VT, RT], Iterable[Instance[KT, DT, VT, RT]]]

class OnnxClassifier(AbstractClassifier[IT, KT, DT, VT, RT, LT, LMT, PMT], Generic[IT, KT, DT, VT, RT, LT, LMT, PMT]):
    def __init__