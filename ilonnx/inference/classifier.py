from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from os import PathLike

import instancelib as il
from instancelib.typehints.typevars import LT
import numpy as np
import onnxruntime as ort
from sklearn.base import ClassifierMixin

from .translators import IdentityPreProcessor, OnnxSeqMapDecoder, OnnxDecoder, OnnxVectorClassLabelDecoder, PreProcessor
from .utils import model_configuration, model_details

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

    @classmethod
    def build_data(cls, 
                   model_location: "PathLike[str]",
                   classes: Optional[Sequence[LT]] = None,
                   storage_location: "Optional[PathLike[str]]"=None, 
                   filename: "Optional[PathLike[str]]"=None, 
                   ints_as_str: bool = False,
                   ) -> il.AbstractClassifier[Any, Any, Any, Any, Any, LT, np.ndarray, np.ndarray]:
        onnx = cls.build_model(model_location)
        il_model = il.SkLearnDataClassifier.build_from_model(onnx, classes, storage_location, filename, ints_as_str)
        return il_model

    @classmethod
    def build_vector(cls, 
                     model_location: "PathLike[str]",
                     classes: Optional[Sequence[LT]] = None,
                     storage_location: "Optional[PathLike[str]]"=None, 
                     filename: "Optional[PathLike[str]]"=None, 
                     ints_as_str: bool = False,
                     ) -> il.AbstractClassifier[Any, Any, Any, Any, Any, LT, np.ndarray, np.ndarray]:
        onnx = cls.build_model(model_location)
        il_model = il.SkLearnVectorClassifier.build_from_model(onnx, classes, storage_location, filename, ints_as_str)
        return il_model