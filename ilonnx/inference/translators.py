from abc import ABC, abstractmethod
from typing import Any, Generic, Mapping, Sequence, TypeVar

import numpy as np
import onnxruntime as ort

from ilonnx.inference.base import OnnxDType, OnnxTensor, OnnxType, OnnxVariable
from instancelib.utils.numpy import to_bicolumn_proba

from .utils import sigmoid, to_matrix

_T = TypeVar("_T")

class OnnxDecoder(ABC):

    def __init__(self, 
                 model_input: OnnxVariable,
                 model_output: OnnxVariable, 
                 *_, **__) -> None:
        self.model_input = model_input
        self.model_output = model_output

    def run_model(self, session: ort.InferenceSession, input_value: Any) -> Any:
        pred_onnx = session.run([self.model_output.name], {self.model_input.name: input_value})[0]
        return pred_onnx
    
    def __call__(self, 
                 session:  ort.InferenceSession, 
                 input_value: Any, *args, **kwargs) -> np.ndarray:
        return self.run_model(session, input_value)

class ProbaPostProcessor(ABC):

    def __init__(self, *_, **__) -> None:
        pass

    @abstractmethod
    def __call__(self, input_value: np.ndarray, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

class PreProcessor(ABC):

    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def __call__(self, input_value: Any, *args, **kwargs) -> Any:
        raise NotImplementedError

class IdentityPreProcessor(PreProcessor):

    def __call__(self, input_value: _T, *args, **kwargs) -> _T:
        return input_value

class IdentityPostProcessor(ProbaPostProcessor):
    def __call__(self, input_value: np.ndarray, *args, **kwargs) -> np.ndarray:
        return input_value

class SigmoidPostProcessor(ProbaPostProcessor):
    def __call__(self, input_value: np.ndarray, *args, **kwargs) -> np.ndarray:
        sigmoided = sigmoid(input_value)
        return sigmoided

class OnnxTensorDecoder(OnnxDecoder):
    def __init__(self, model_input: OnnxVariable,
                       model_output: OnnxVariable,
                       proba_post_processor: ProbaPostProcessor = IdentityPostProcessor(),
                       *_, **__) -> None:
        super().__init__(model_input, model_output)
        self.proba_post_processor = proba_post_processor

    def __call__(self, 
                 session: ort.InferenceSession, 
                 input_value: np.ndarray) -> np.ndarray:
        """Return the predicted classes for each input

        Parameters
        ----------
        input_value : np.ndarray
            A feature matrix or another form raw input data that can
            be fed to the ONNX model

        Returns
        -------
        np.ndarray
            A tensor that contains the predicted classes
        """
        pred_onnx: np.ndarray = self.run_model(session, input_value)
        pred_processed = to_bicolumn_proba(self.proba_post_processor(pred_onnx))
        return pred_processed

class OnnxSeqMapDecoder(OnnxDecoder):
    def __call__(self, session: ort.InferenceSession, input_value: Any, *args, **kwargs) -> np.ndarray:
        """Return the predicted class probabilities for each input

        Parameters
        ----------
        X : np.ndarray
            A feature matrix or another form raw input data that can
            be fed to the ONNX model

        Returns
        -------
        np.ndarray
            A probability matrix
        """        
        pred_onnx = self.run_model(session, input_value)
        prob_vec = to_matrix(pred_onnx)
        return prob_vec

    
class OnnxVectorClassLabelDecoder(OnnxDecoder):
    def __call__(self, 
                 session: ort.InferenceSession, 
                 input_value: np.ndarray) -> np.ndarray:
        """Return the predicted classes for each input

        Parameters
        ----------
        X : np.ndarray
            A feature matrix or another form raw input data that can
            be fed to the ONNX model

        Returns
        -------
        np.ndarray
            A tensor that contains the predicted classes
        """              
        pred_onnx: np.ndarray = self.run_model(session, input_value)
        return pred_onnx

class OnnxThresholdPredictionDecoder(OnnxTensorDecoder):
    def __init__(self, 
                 model_input: OnnxVariable, 
                 model_output: OnnxVariable, 
                 threshold: float = 0.5,
                 proba_post_processor: ProbaPostProcessor = IdentityPostProcessor(),
                 *_, **__
                 ) -> None:
        super().__init__(model_input, model_output, proba_post_processor=proba_post_processor)
        self.threshold = threshold

    def __call__(self, session: ort.InferenceSession, input_value: np.ndarray) -> np.ndarray:
        proba_result = super().__call__(session, input_value)
        pred_binary = proba_result >= self.threshold
        pred_int = pred_binary.astype(np.int64)
        return pred_int


class OnnxTensorEncoder(PreProcessor):
    def __init__(self, model_input: OnnxVariable):
        self.model_input = model_input
        assert isinstance(self.model_input.vartype, OnnxTensor)
        self.dtype = self.model_input.vartype.dtype

    def __call__(self, input_value: np.ndarray, *args, **kwargs) -> np.ndarray:
        if self.dtype == OnnxDType.FLOAT:
            conv_input = input_value.astype(np.float32)
        elif self.dtype == OnnxDType.DOUBLE:
            conv_input = input_value.astype(np.double)
        elif self.dtype == OnnxDType.INT64:
            conv_input = input_value.astype(np.int64)
        else:
            conv_input = input_value
        return conv_input