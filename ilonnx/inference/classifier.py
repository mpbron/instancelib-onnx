from typing import Any, Mapping, Optional, Sequence

from os import PathLike

import instancelib as il
from instancelib.typehints.typevars import LT
import numpy as np
import onnxruntime as rt
from sklearn.base import ClassifierMixin

class OnnxClassifier(ClassifierMixin):
    """Adapter Class for ONNX models. 
    This class loads an ONNX model and provides an interface that conforms to the scikit-learn classifier API.
    """    

    def __init__(self,  model_location: "PathLike[str]") -> None:
        """Initialize the model. 
        The model stored in the argument's location is loaded.

        Parameters
        ----------
        model_location : PathLike[str]
            The location of the model
        """        
        self._sess = rt.InferenceSession(model_location)
        
        self._input_name = self._sess.get_inputs()[0].name
        self._label_name = self._sess.get_outputs()[0].name
        self._proba_name = self._sess.get_outputs()[1].name

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
        conv_input = X.astype(np.float32)        
        pred_onnx: np.ndarray = self._sess.run([self._label_name], {self._input_name: conv_input})[0]
        return pred_onnx

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
        conv_input = X.astype(np.float32)
        pred_onnx = self._sess.run([self._proba_name],{self._input_name: conv_input})[0]
        prob_vec = self._to_matrix(pred_onnx)
        return prob_vec

    def _to_matrix(self, y: Sequence[Mapping[Any, float]]) -> np.ndarray:
        """Converts ONNX output to standard scikit-learn ``predict_proba`` 

        Parameters
        ----------
        y : Sequence[Mapping[Any, float]]
            A sequence of mappings of labels to floats

        Returns
        -------
        np.ndarray
            A probability matrix of shape (n_inputs, n_labels)
        """        
        if y:
            result_matrix = np.zeros(
                shape=(len(y), len(y[0])), 
                dtype=np.float32)
            
            for i, row in enumerate(y):
                for (lbl_idx, (lbl, proba)) in enumerate(row.items()):
                    if isinstance(lbl, int):
                        j = lbl
                    else:
                        j = lbl_idx
                    result_matrix[i,j] = proba
            return result_matrix
        return np.zeros(shape=(0,0), dtype=np.float32)

    @classmethod
    def build_data(cls, 
                   model_location: "PathLike[str]",
                   classes: Optional[Sequence[LT]] = None,
                   storage_location: "Optional[PathLike[str]]"=None, 
                   filename: "Optional[PathLike[str]]"=None, 
                   ints_as_str: bool = False,
                   ) -> il.AbstractClassifier[Any, Any, Any, Any, Any, LT, np.ndarray, np.ndarray]:
        onnx = cls(model_location)
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
        onnx = cls(model_location)
        il_model = il.SkLearnVectorClassifier.build_from_model(onnx, classes, storage_location, filename, ints_as_str)
        return il_model
