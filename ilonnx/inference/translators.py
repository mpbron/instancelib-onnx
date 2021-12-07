from abc import ABC, abstractmethod
from typing import Any, Generic, Mapping, Sequence, TypeVar

import numpy as np
import onnxruntime as rt

from .utils import sigmoid, to_matrix

_T = TypeVar("_T")

class OnnxTranslator(ABC):
    onnx_session: rt.InferenceSession

    @abstractmethod
    def __call__(self, input_value: Any, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

class ProbaPostProcessor(ABC):

    @abstractmethod
    def __call__(self, input_value: np.ndarray, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

class PreProcessor(ABC):

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




class OnnxSeqMapDecoder(OnnxTranslator):

    def __init__(self, onnx_session: rt.InferenceSession, 
                       input_field: str, 
                       proba_field: str) -> None:
        self.onnx_session = onnx_session
        self.input_field = input_field
        self.proba_field = proba_field

    def __call__(self, input_value: np.ndarray) -> np.ndarray:
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
        conv_input = input_value.astype(np.float32)
        pred_onnx = self.onnx_session.run([self.proba_field], {self.input_field: conv_input})[0]
        prob_vec = to_matrix(pred_onnx)
        return prob_vec

    
class OnnxVectorClassLabelDecoder(OnnxTranslator):
    def __init__(self, onnx_session: rt.InferenceSession, 
                       input_field: str, 
                       pred_field: str) -> None:
        self.onnx_session = onnx_session
        self.input_field = input_field
        self.pred_field = pred_field

    def __call__(self, input_value: np.ndarray) -> np.ndarray:
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
        conv_input = input_value.astype(np.float32)        
        pred_onnx: np.ndarray = self.onnx_session.run([self.pred_field], {self.input_field: conv_input})[0]
        return pred_onnx

class OnnxThresholdPredictionDecoder(OnnxTranslator):
    def __init__(self, onnx_session: rt.InferenceSession, 
                       input_field: str, 
                       proba_field: str,
                       proba_post_processor: ProbaPostProcessor = IdentityPostProcessor(),
                       threshold: float = 0.5) -> None:
        self.onnx_session = onnx_session
        self.input_field = input_field
        self.proba_field = proba_field
        self.proba_post_processor = proba_post_processor
        self.threshold = threshold

    def __call__(self, input_value: np.ndarray) -> np.ndarray:
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
        conv_input = input_value.astype(np.float32)        
        pred_onnx: np.ndarray = self._sess.run([self.proba_field], {self.input_field: conv_input})[0]
        pred_processed  = self.proba_post_processor(pred_onnx)
        pred_binary = pred_processed >= self.threshold
        pred_int = pred_binary.astype(np.int64)
        return pred_int

class OnnxTensorDecoder(OnnxTranslator):
    def __init__(self, onnx_session: rt.InferenceSession, 
                       input_field: str, 
                       proba_field: str,
                       proba_post_processor: ProbaPostProcessor = IdentityPostProcessor()) -> None:
        self.onnx_session = onnx_session
        self.input_field = input_field
        self.proba_field = proba_field
        self.proba_post_procesor = proba_post_processor

    def __call__(self, input_value: np.ndarray) -> np.ndarray:
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
        conv_input = input_value.astype(np.float32)        
        pred_onnx: np.ndarray = self._sess.run([self.proba_field], {self.input_field: conv_input})[0]
        pred_processed  = self.proba_post_processor(pred_onnx)
        return pred_processed