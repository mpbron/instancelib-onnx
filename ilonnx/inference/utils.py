from typing import Any, Mapping, Sequence
import numpy as np

import onnxruntime as ort

def to_matrix(y: Sequence[Mapping[Any, float]]) -> np.ndarray:
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

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-z))

def model_details(session: ort.InferenceSession) -> None:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    print("Inputs\n======")
    for inp in inputs:
        print(f"{inp.name} :: {inp.type} ({inp.shape})")
    print("Outputs\n=======")
    for out in outputs:
        print(f"{out.name} :: {out.type} ({out.shape})")
    metadata = session.get_modelmeta()
    print(metadata)