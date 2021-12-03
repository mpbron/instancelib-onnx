from __future__ import annotations

from typing import Any, Optional, Sequence

from os import PathLike

import instancelib as il
from instancelib.typehints.typevars import LT
import numpy as np
import onnxruntime as rt
from sklearn.base import ClassifierMixin

from .translators import OnnxSeqMapDecoder, OnnxTranslator, OnnxVectorClassLabelDecoder
from .utils import model_details

class OnnxClassifier(ClassifierMixin):
    """Adapter Class for ONNX models. 
    This class loads an ONNX model and provides an interface that conforms to the scikit-learn classifier API.
    """    

    def __init__(self, pred_decoder: OnnxTranslator, 
                       proba_decoder: OnnxTranslator) -> None:
        """Initialize the model. 
        The model stored in the argument's location is loaded.

        Parameters
        ----------
        model_location : PathLike[str]
            The location of the model
        """        
        self.pred_decoder = pred_decoder
        self.proba_decoder = proba_decoder

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

    def predict(self, X: np.ndarray) -> np.ndarray:
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
        translation = self.pred_decoder(X)        
        return translation

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
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
        translation = self.pred_decoder(X)        
        return translation

    @classmethod
    def build_model(cls, model_location: "PathLike[str]") -> OnnxClassifier:
        session = rt.InferenceSession(model_location)
        model_details(session)
        input_field = session.get_inputs()[0].name
        pred_field = session.get_outputs()[0].name
        proba_field = session.get_outputs()[1].name
        pred_decoder = OnnxVectorClassLabelDecoder(session, input_field, pred_field)
        proba_decoder = OnnxSeqMapDecoder(session, input_field, proba_field)
        return cls(pred_decoder, proba_decoder)
        
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