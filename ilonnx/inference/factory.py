from os import PathLike
from typing import Any, Mapping, Optional, Sequence, Union
from instancelib.typehints.typevars import LT
from instancelib.utils.func import seq_or_map_to_map

import onnxruntime as ort
from ilonnx.factory.base import AbstractBuilder, ObjectFactory
from ilonnx.inference.classifier import OnnxClassifier
from instancelib.machinelearning.sklearn import SkLearnClassifier
from ilonnx.inference.encoder import OnnxLabelEncoder
from ilonnx.inference.translators import (IdentityPostProcessor,
                                          OnnxSeqMapDecoder, OnnxTensorDecoder,
                                          OnnxTensorEncoder,
                                          OnnxThresholdPredictionDecoder,
                                          OnnxVectorClassLabelDecoder,
                                          SigmoidPostProcessor)
from ilonnx.inference.utils import model_configuration
import instancelib as il

from .base import (OnnxBaseType, OnnxComponent, OnnxDType, OnnxMap,
                   OnnxSequence, OnnxTensor, OnnxTypeEnum, OnnxValueType,
                   OnnxVariable, PostProcessorType)

SEQMAPINTFLOAT = OnnxSequence(
    OnnxMap(OnnxBaseType(OnnxValueType.INT64), OnnxTensor(OnnxDType.FLOAT)))

INT_TENSOR = OnnxTensor(OnnxDType.INT64)
FLOAT_TENSOR = OnnxTensor(OnnxDType.FLOAT)
DOUBLE_TENSOR = OnnxTensor(OnnxDType.DOUBLE)
STRING_TENSOR = OnnxTensor(OnnxDType.STRING)



class ClassifierBuilder(AbstractBuilder):
    def __call__(self, 
                 session: ort.InferenceSession,
                 inputs: Sequence[OnnxVariable], 
                 outputs: Sequence[OnnxVariable],
                 input_name: Optional[str] = None,
                 **kwargs):
        model_input = input_selector(inputs, input_name)
        if model_input is None:
            raise ValueError(f"No suitable input found in {inputs}")
        preprocessor = self._factory.create(
            OnnxComponent.PREPROCESSOR, model_input=model_input, **kwargs)
        proba_decoder = self._factory.create(
            OnnxComponent.PROBA_DECODER, model_input=model_input, outputs=outputs, **kwargs)
        pred_decoder = self._factory.create(
            OnnxComponent.PRED_DECODER, model_input=model_input, outputs=outputs, **kwargs)
        return OnnxClassifier(session, preprocessor, pred_decoder, proba_decoder)


class InputEncoderBuilder(AbstractBuilder):
    def __call__(self, model_input: Sequence[OnnxVariable], **kwargs):
        input_type = model_input.vartype.otype
        build_key = (OnnxComponent.PREPROCESSOR, input_type) 
        return self._factory.create(build_key, model_input=model_input)

def is_proba_output(var: OnnxVariable) -> bool:
    if "proba" in var.name:
        return True
    vartype = var.vartype
    if isinstance(vartype, OnnxTensor):
        return vartype.dtype in [OnnxDType.FLOAT, OnnxDType.DOUBLE]
    if isinstance(vartype, OnnxSequence):
        return vartype.item_type == SEQMAPINTFLOAT
    return False

def is_pred_output(var: OnnxVariable) -> bool:
    if var.name in ["output_label"]:
        return True
    vartype = var.vartype
    if isinstance(vartype, OnnxTensor):
        return vartype.dtype in [OnnxDType.INT64]
    return False

def input_selector(variables: Sequence[OnnxVariable], 
                   input_name: Optional[str] = None) -> Optional[OnnxVariable]:
    if input_name is not None:
        matches = (var for var in variables if var.name == input_name)
        return next(matches, None)
    if len(variables) == 1:
        return variables[0]
    return None

def proba_output_selector(variables: Sequence[OnnxVariable], 
                        proba_output_name: Optional[str] = None, 
                        **kwargs) -> Optional[OnnxVariable]:
    if proba_output_name is not None:
        matches = (var for var in variables if var.name == proba_output_name)
        return next(matches, None)
    if len(variables) == 1:
        return variables[0]
    matches = [var for var in variables if is_proba_output(var)]
    if matches:
        return matches[0]
    return None

def pred_output_selector(variables: Sequence[OnnxVariable], 
                         pred_output_name: Optional[str] = None, 
                         proba_output_name: Optional[str] = None,
                         **kwargs) -> Optional[OnnxVariable]:
    if pred_output_name is not None:
        matches = (var for var in variables if var.name == pred_output_name)
        return next(matches, None)
    if len(variables) == 1:
        return variables[0]
    matches = [var for var in variables if is_pred_output(var)]
    if matches:
        return matches[0]
    return proba_output_selector(variables, proba_output_name)

class PredDecoderBuilder(AbstractBuilder):
    def __call__(self, model_input: OnnxVariable, 
                       outputs: Sequence[OnnxVariable],
                       pred_output_name: Optional[str] = None,
                       proba_output_name: Optional[str] = None, 
                       **kwargs):
        pred_output = pred_output_selector(outputs, 
                                           pred_output_name, 
                                           proba_output_name)
        if pred_output is not None:
            build_key = (OnnxComponent.PRED_DECODER, pred_output.vartype)
            return self._factory.create(build_key, 
                                 model_input=model_input, 
                                 model_output=pred_output, **kwargs)
        return ValueError(f"No suitable class label predction output found in {outputs}")
class ProbaDecoderBuilder(AbstractBuilder):
    def __call__(self, model_input: OnnxVariable, 
                       outputs: Sequence[OnnxVariable],
                       proba_output_name: Optional[str] = None, 
                       **kwargs):
        proba_output = proba_output_selector(outputs, proba_output_name)
        if proba_output is not None:
            build_key = (OnnxComponent.PROBA_DECODER, proba_output.vartype)
            return self._factory.create(build_key, 
                                 model_input=model_input, 
                                 model_output=proba_output, **kwargs)
        raise ValueError(f"No suitable output found in {outputs}")
        

class TensorProbaDecoderBuilder(AbstractBuilder):
    def __call__(self, model_input: OnnxVariable, 
                       model_output: OnnxVariable,
                       post_processor = PostProcessorType.IDENTITY,
                       **kwargs):
        build_key = (OnnxComponent.POST_PROCESSOR, post_processor)
        proba_post_processor = self._factory.create(build_key, **kwargs)
        return OnnxTensorDecoder(model_input, model_output, proba_post_processor)

class PredThresholdDecoderBuilder(AbstractBuilder):
    def __call__(self, model_input: OnnxVariable, 
                       model_output: OnnxVariable,
                       threshold: float = 0.5,
                       post_processor = PostProcessorType.IDENTITY,
                       **kwargs):
        build_key = (OnnxComponent.POST_PROCESSOR, post_processor)
        proba_post_processor = self._factory.create(build_key, **kwargs)
        return OnnxThresholdPredictionDecoder(model_input, 
                                              model_output, 
                                              threshold,
                                              proba_post_processor)

class OnnxFactory(ObjectFactory):
    def __init__(self) -> None:
        super().__init__()
        self.register_builder(OnnxComponent.CLASSIFIER, ClassifierBuilder())
        self.register_builder(OnnxComponent.PREPROCESSOR, InputEncoderBuilder())
        self.register_builder(OnnxComponent.PROBA_DECODER, ProbaDecoderBuilder())
        self.register_builder(OnnxComponent.PRED_DECODER, PredDecoderBuilder())
        # Pre processors
        self.register_constructor((OnnxComponent.PREPROCESSOR, 
                                   OnnxTypeEnum.TENSOR), OnnxTensorEncoder)

        # Proba decoders
        self.register_builder((OnnxComponent.PROBA_DECODER, 
                                   FLOAT_TENSOR), TensorProbaDecoderBuilder())
        self.register_constructor((OnnxComponent.PROBA_DECODER, 
                                   SEQMAPINTFLOAT), OnnxSeqMapDecoder)
        
        # Prediction decoders
        self.register_builder((OnnxComponent.PRED_DECODER, FLOAT_TENSOR), PredThresholdDecoderBuilder())
        self.register_builder((OnnxComponent.PRED_DECODER, DOUBLE_TENSOR), PredThresholdDecoderBuilder())
        self.register_constructor((OnnxComponent.PRED_DECODER, INT_TENSOR), OnnxVectorClassLabelDecoder)
        self.register_constructor((OnnxComponent.PRED_DECODER, STRING_TENSOR), OnnxVectorClassLabelDecoder)
        
        # Proba post processors
        self.register_constructor((OnnxComponent.POST_PROCESSOR, 
                                   PostProcessorType.IDENTITY), IdentityPostProcessor)
        self.register_constructor((OnnxComponent.POST_PROCESSOR,
                                   PostProcessorType.SIGMOID), SigmoidPostProcessor)

    def build_model(self,
                    model_location: PathLike["str"], 
                    post_processor = PostProcessorType.IDENTITY, 
                    **kwargs) -> OnnxClassifier:
        session = ort.InferenceSession(model_location)
        configuration = model_configuration(session)
        classifier = self.create(OnnxComponent.CLASSIFIER, 
                            session=session, 
                            post_processor=post_processor,
                            **configuration, **kwargs)
        return classifier

    def build_vector_model(self,
                           model_location: PathLike["str"],
                           classes : Union[Sequence[LT], Mapping[int, LT]],
                           post_processing = PostProcessorType.IDENTITY, 
                           **kwargs) -> SkLearnClassifier[Any, Any, Any, Any, LT]:
        onnx_model = self.build_model(model_location, post_processing, **kwargs)
        label_mapping = seq_or_map_to_map(classes)
        encoder = OnnxLabelEncoder.from_inv(label_mapping)
        ilmodel = il.SkLearnVectorClassifier(onnx_model, encoder)
        return ilmodel

    def build_data_model(self,
                         model_location: PathLike["str"],
                         classes : Optional[Sequence[Any]] = None,
                         post_processing = PostProcessorType.IDENTITY, 
                         **kwargs) -> SkLearnClassifier[Any, Any, Any, Any, LT]:
        onnx_model = self.build_model(model_location, post_processing, **kwargs)
        label_mapping = seq_or_map_to_map(classes)
        encoder = OnnxLabelEncoder.from_inv(label_mapping)
        ilmodel = il.SkLearnDataClassifier(onnx_model, encoder)
        return ilmodel
