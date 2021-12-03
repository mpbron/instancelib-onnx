from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

import numpy as np
import onnxruntime as rt

from .utils import sigmoid, to_matrix

class OnnxTranslator(ABC):
    onnx_session: rt.InferenceSession

    @abstractmethod
    def __call__(self, input_value: Any, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError




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
                       threshold: float = 0.5) -> None:
        self.onnx_session = onnx_session
        self.input_field = input_field
        self.proba_field = proba_field
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
        pred_binary = pred_onnx >= self.threshold
        pred_int = pred_binary.astype(np.int64)
        return pred_int

class OnnxIdentityDecoder(OnnxTranslator):
    def __init__(self, onnx_session: rt.InferenceSession, 
                       input_field: str, 
                       proba_field: str,
                       threshold: float = 0.5) -> None:
        self.onnx_session = onnx_session
        self.input_field = input_field
        self.proba_field = proba_field
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
        return pred_onnx


class OnnxPostSigmoidDecoder(OnnxIdentityDecoder):
    def __call__(self, input_value: np.ndarray) -> np.ndarray:
        prediction = super().__call__(input_value)
        sigmoided = sigmoid(prediction)
        return sigmoided