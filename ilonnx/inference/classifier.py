from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from os import PathLike

import instancelib as il
from instancelib.typehints.typevars import LT
import numpy as np
import onnxruntime as ort
from sklearn.base import ClassifierMixin

from ilonnx.inference.base import OnnxVariable, get_shape

from .translators import IdentityPreProcessor, OnnxSeqMapDecoder, OnnxDecoder, OnnxVectorClassLabelDecoder, PreProcessor
from .utils import model_configuration, model_details
from .parsing import pOnnxType

class OnnxClassifier(ClassifierMixin):
    """Adapter Class for ONNX models. 
    This class loads an ONNX model and provides an interface that conforms to the scikit-learn classifier API.
    """    

    def __init__(self, session: ort.InferenceSession,
                       preprocessor: PreProcessor,
                       pred_decoder: OnnxDecoder, 
                       proba_decoder: OnnxDecoder
                ) -> None:
        """Initialize the model. 
        The model stored in the argument's location is loaded.

        Parameters
        ----------
        model_location : PathLike[str]
            The location of the model
        """        
        self.preprocessor = preprocessor
        self.pred_decoder = pred_decoder
        self.proba_decoder = proba_decoder
        self.session = session
        

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fitting a model is not supported. Inference only!
        This method will not do anything and return itself.

        Parameters
        ----------
        X : np.ndarray
            A feature matrix or another form raw input data that can
            be fed to the ONNX model
        y : np.ndarray
            The target class labels
        """        
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Return the predicted classes for each input

        Parameters
        ----------
        X : Any
            A feature matrix or another form raw input data that can
            be fed to the ONNX model

        Returns
        -------
        np.ndarray
            A tensor that contains the predicted classes
        """
        encoded_X = self.preprocessor(X)
        Y = self.pred_decoder(self.session, encoded_X)
        return Y


    def predict_proba(self, X: Any) -> np.ndarray:
        """Return the predicted class probabilities for each input

        Parameters
        ----------
        X : Any
            A feature matrix or another form raw input data that can
            be fed to the ONNX model

        Returns
        -------
        np.ndarray
            A probability matrix
        """
        encoded_X = self.preprocessor(X)        
        Y = self.proba_decoder(self.session, encoded_X)
        return Y

    def __repr__(self) -> str:
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        parsed_inputs = [OnnxVariable(var.name, pOnnxType.parse(var.type), get_shape(var)) for var in inputs]
        parsed_outputs =[OnnxVariable(var.name, pOnnxType.parse(var.type), get_shape(var)) for var in outputs]
        metadata = self.session.get_modelmeta()
        result = (f"OnnxClassifier("
                  f"preprocessor={repr(self.preprocessor)}, "
                  f"pred_decoder={repr(self.pred_decoder)}, "
                  f"proba_decoder={repr(self.proba_decoder)}, "
                  "onnx_session={"
                  f"inputs={repr(parsed_inputs)}, "
                  f"outputs={repr(parsed_outputs)}, "
                  "metadata={"
                  f"producer={metadata.producer_name}, "
                  f"domain={metadata.domain}, "
                  f"graph_name={metadata.graph_name}"
                  "}})")
        return result

    def __str__(self) -> str:
        return self.__repr__()