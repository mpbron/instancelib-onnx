from abc import ABC, abstractmethod
import functools
from typing import Any, FrozenSet, Generic, Iterable, Iterator, Sequence, Tuple, TypeVar, Union
from instancelib.labels.base import LabelProvider
from instancelib.machinelearning.base import AbstractClassifier
from instancelib.typehints.typevars import  KT, DT, VT, RT, LT, LMT, PMT
from instancelib.machinelearning.featurematrix import FeatureMatrix
from instancelib import InstanceProvider, Instance
from instancelib.labels import LabelEncoder
from os import PathLike
from instancelib.utils.func import zip_chain
from instancelib.utils.chunks import divide_iterable_in_lists

import itertools
import numpy as np
import onnxruntime as rt

IT = TypeVar("IT",
             bound="Instance[Any,Any,Any,Any]", covariant=True)

InstanceInput = Union[InstanceProvider[IT, KT, DT,
                                       VT, RT], Iterable[Instance[KT, DT, VT, RT]]]


class OnnxClassifier(AbstractClassifier[IT, KT, DT, VT, RT, LT, 
                                        "np.ndarray[Any]", "np.ndarray[Any]"], Generic[IT, KT, DT, VT, RT, LT]):

    def __init__(self, 
                 model_location: "PathLike[str]", 
                 label_encoder: LabelEncoder[LT, "np.ndarray[Any]", "np.ndarray[Any]", "np.ndarray[Any]"]) -> None:
        self._sess = rt.InferenceSession(model_location)
        
        self._input_name = self._sess.get_inputs()[0].name
        self._label_name = self._sess.get_outputs()[0].name
        
        self._proba_name = self._sess.get_outputs()[1].name
        self.encoder = label_encoder

    def get_label_column_index(self, label: LT) -> int:
        return self.encoder.get_label_column_index(label)

    def fit_instances(self, 
                      instances: Iterable[Instance[KT, DT, VT, RT]], 
                      labels: Iterable[Iterable[LT]]) -> None:
        raise NotImplementedError("We cannot fit ONNX models")

    def fit_provider(self, 
                     provider: InstanceProvider[IT, KT, DT, VT, RT], 
                     labels: LabelProvider[KT, LT], 
                     batch_size: int = 200) -> None:
        raise NotImplementedError("We cannot fit ONNX models")

    def _get_probas(self, matrix: FeatureMatrix[KT]) -> Tuple[Sequence[KT], np.ndarray]:
        """Calculate the probability matrix for the current feature matrix
        
        Parameters
        ----------
        matrix : FeatureMatrix[KT]
            The matrix for which we want to know the predictions
        
        Returns
        -------
        Tuple[Sequence[KT], np.ndarray]
            A list of keys and the probability predictions belonging to it
        """
        prob_vec: np.ndarray = self._predict_proba(matrix.matrix)  # type: ignore
        keys = matrix.indices
        return keys, prob_vec

    def _get_preds(self, matrix: FeatureMatrix[KT]) -> Tuple[Sequence[KT], Sequence[FrozenSet[LT]]]:
        """Predict the labels for the current feature matrix
        
        Parameters
        ----------
        matrix : FeatureMatrix[KT]
            The matrix for which we want to know the predictions
        
        Returns
        -------
        Tuple[Sequence[KT], Sequence[FrozenSet[LT]]]
            A list of keys and the predictions belonging to it
        """
        pred_vec: np.ndarray = self._predict(matrix.matrix)
        keys = matrix.indices
        labels = self.encoder.decode_matrix(pred_vec)
        return keys, labels


    def _get_probas(self, matrix: FeatureMatrix[KT]) -> Tuple[Sequence[KT], np.ndarray]:
        """Calculate the probability matrix for the current feature matrix
        
        Parameters
        ----------
        matrix : FeatureMatrix[KT]
            The matrix for which we want to know the predictions
        
        Returns
        -------
        Tuple[Sequence[KT], np.ndarray]
            A list of keys and the probability predictions belonging to it
        """
        prob_vec: np.ndarray = self._predict_proba(matrix.matrix.tolist())  # type: ignore
        keys = matrix.indices
        return keys, prob_vec

    def _decode_proba_matrix(self, 
                             keys: Sequence[KT], 
                             y_matrix: np.ndarray) -> Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]:
        y_labels = self.encoder.decode_proba_matrix(y_matrix)
        zipped = list(zip(keys, y_labels)) 
        return zipped

    def predict_proba_provider_raw(self, 
                                   provider: InstanceProvider[IT, KT, Any, np.ndarray, Any],
                                   batch_size: int = 200) -> Iterator[Tuple[Sequence[KT], np.ndarray]]:
        matrices = FeatureMatrix[KT].generator_from_provider(provider, batch_size)
        preds = map(self._get_probas, matrices)
        yield from preds

    def predict_provider(self, 
                         provider: InstanceProvider[IT, KT, Any, np.ndarray, Any], 
                         batch_size: int = 200) -> Sequence[Tuple[KT, FrozenSet[LT]]]:
        matrices = FeatureMatrix[KT].generator_from_provider(provider, batch_size)
        preds = map(self._get_preds, matrices)
        results = list(zip_chain(preds))
        return results

    def predict_proba_provider(self, 
                               provider: InstanceProvider[IT, KT, DT, VT, Any], 
                               batch_size: int = 200) -> Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]:
        preds = self.predict_proba_provider_raw(provider, batch_size)
        decoded_probas = itertools.starmap(self._decode_proba_matrix, preds)
        chained = list(itertools.chain.from_iterable(decoded_probas))
        return chained
    
    def predict_proba_instances_raw(self, instances: Iterable[Instance[KT, DT, VT, RT]], batch_size: int = 200) -> Iterator[Tuple[Sequence[KT], PMT]]:
        return super().predict_proba_instances_raw(instances, batch_size=batch_size)
    
    @property
    def fitted(self) -> bool:
        return super().fitted

    def set_target_labels(self, labels: Iterable[LT]) -> None:
        return super().set_target_labels(labels)
    
    def _predict(self, input: "Sequence[np.ndarray[Any]]") -> np.ndarray:        
        pred_onnx: "np.ndarray[Any]" = self._sess.run([self._label_name], [{self._input_name: input}])[0]
        return pred_onnx

    def _predict_proba(self, input: "Sequence[np.ndarray[Any]]") -> np.ndarray:
        pred_onnx: "np.ndarray[float]" = self._sess.run([self._proba_name],{self._input_name: input})[0]
        return pred_onnx

    def predict_proba_instances(self, 
                                instances: Iterable[Instance[KT, DT, VT, Any]],
                                batch_size: int = 200
                                ) -> Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]:
        
        batches = divide_iterable_in_lists(instances, batch_size)
        processed = map(self._pred_proba_ins_batch, batches)
        combined: Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]] = functools.reduce(
            operator.concat, processed) # type: ignore
        return combined

    def predict_instances(self, 
                          instances: Iterable[Instance[KT, DT, VT, Any]],
                          batch_size: int = 200) -> Sequence[Tuple[KT, FrozenSet[LT]]]:        
        batches = divide_iterable_in_lists(instances, batch_size)
        results = map(self._pred_ins_batch, batches)
        concatenated: Sequence[Tuple[KT, FrozenSet[LT]]] = functools.reduce(
            lambda a,b: operator.concat(a,b), results) # type: ignore
        return concatenated