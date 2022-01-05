from typing import FrozenSet, Generic, Sequence
from instancelib.labels.encoder import DictionaryEncoder, MultilabelDictionaryEncoder
from instancelib.typehints.typevars import LT
import numpy as np

class OnnxLabelEncoder(MultilabelDictionaryEncoder[LT], Generic[LT]):
    def decode_matrix(self, matrix: np.ndarray) -> Sequence[FrozenSet[LT]]:
        if len(matrix.shape) == 1:
            return super(MultilabelDictionaryEncoder, self).decode_matrix(matrix)
        return super().decode_matrix(matrix)

    def decode_vector(self, vector: np.ndarray) -> FrozenSet[LT]:
        if vector.shape[0] > 1:
            return super(MultilabelDictionaryEncoder, self).decode_vector(vector)
        return super().decode_vector(vector)