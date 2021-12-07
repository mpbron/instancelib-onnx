from typing import Any
from ilonnx.factory.base import ObjectFactory, AbstractBuilder
from .base import OnnxComponent, OnnxDType, OnnxSeqMapVariable, OnnxTensorVariable, OnnxValueType, OnnxType, get_dimensions, get_shape
from .parsing import pOnnxVar
class Variable(AbstractBuilder):
    def __call__(self, vinfo: Any, **kwargs):
        parseResult: OnnxType = pOnnxVar.parse(vinfo.type)
        return self._factory(parseResult.otype,
                             parsed_type = parseResult, 
                             variable_info = vinfo,
                             **kwargs)        
        

class TensorVariableBuilder(AbstractBuilder):
    def __call__(self, 
                 parsed_type: OnnxTensorVariable,
                 variable_info: Any,
                 **kwargs):
        retur

class SeqMapVariableBuilder(AbstractBuilder):
    def __call__(self, vinfo: Any, **kwargs):
        return OnnxSeqMapVariable(vinfo.name, OnnxValueType.SEQMAP,)

class OnnxFactory(ObjectFactory):
    def __init__(self) -> None:
        super().__init__()
        # self.register_builder(Component.CLASSIFIER, ClassifierBuilder())
        # self.register_builder(ML.Task.BINARY, BinaryClassificationBuilder())
        # self.register_builder(ML.Task.MULTICLASS, MulticlassBuilder())
        # self.register_builder(ML.Task.MULTILABEL, MultilabelBuilder())
        # self.register_builder(ML.MulticlassMethod.ONE_VS_REST, OneVsRestBuilder())
        