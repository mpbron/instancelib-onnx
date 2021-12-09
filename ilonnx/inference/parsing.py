import re

from parsec import generate, regex, separated, string

from .base import (OnnxBaseType, OnnxDType, OnnxMap, OnnxSequence, OnnxTensor,
                   OnnxValueType)

spaces = regex(r'\s*', re.MULTILINE)
name = regex(r'[_a-zA-Z][_a-zA-Z0-9]*')

def pBrackets(pFunc):
    return string("(") >> pFunc << string(")")

@generate
def pOnnxType():
    type_str = yield name
    return OnnxValueType(type_str)


@generate
def pOnnxDType():
    dtype_str = yield name
    return OnnxDType(dtype_str)


@generate
def pTensorType():
    yield string("tensor")
    dtype = yield pBrackets(pOnnxDType)
    return OnnxTensor(dtype)

@generate
def pOnnxMapType():
    yield string("map")
    listvars = yield pBrackets(
        separated(
            pOnnxVar,
            string(","), 2, 2))
    return OnnxMap(listvars[0], listvars[1])


@generate
def pOnnxSeqType():
    yield string("seq")
    innervar = yield pBrackets(pOnnxVar)
    return OnnxSequence(innervar)


@generate
def pBaseType():
    vartype = yield pOnnxType
    return OnnxBaseType(vartype)


pOnnxVar = pTensorType | pOnnxMapType | pOnnxSeqType | pBaseType
