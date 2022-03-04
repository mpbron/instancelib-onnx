from os import PathLike
from typing import Any, Mapping, Optional, Sequence, Union
from instancelib.machinelearning.base import AbstractClassifier
from instancelib.typehints.typevars import LT
import numpy as np
from ilonnx.inference.base import PostProcessorType
from ilonnx.inference.factory import OnnxFactory

def build_vector_model(model_location: "PathLike[str]",
                       classes : Optional[Union[Sequence[LT], Mapping[int, LT]]] = None,
                       post_processor = PostProcessorType.IDENTITY,
                       factory: OnnxFactory = OnnxFactory(), 
                       **kwargs) -> AbstractClassifier[Any, Any, Any, np.ndarray, Any, LT, np.ndarray, np.ndarray]:
    """Build an InstanceLib vector model based on an ONNX model file

    Parameters
    ----------
    model_location : PathLike[str]
        The path of the ONNX model file
    classes : Union[Sequence[LT], Mapping[int, LT]]
        The classes that the model uses. 
    post_processor : PostProcessorType, optional
        Some post processing. Not all models need this,
        so by default PostProcessorType.IDENTITY
    factory : OnnxFactory, optional
        The model factory that is used.
        It is not needed to change this, except if you extended this 
        library. 
        By default OnnxFactory()

    Returns
    -------
    AbstractClassifier[Any, Any, Any, np.ndarray, Any, LT, np.ndarray, np.ndarray]
        An InstanceLib wrapped model
    """    
    model = factory.build_vector_model(model_location, classes, post_processor, **kwargs)
    return model

def build_data_model(model_location: "PathLike[str]",
                     classes : Optional[Union[Sequence[LT], Mapping[int, LT]]] = None,
                     post_processor = PostProcessorType.IDENTITY,
                     factory: OnnxFactory = OnnxFactory(), 
                     **kwargs) -> AbstractClassifier[Any, Any, Any, np.ndarray, Any, LT, np.ndarray, np.ndarray]:
    """Build an InstanceLib data model based on an ONNX model file

    Parameters
    ----------
    model_location : PathLike[str]
        The path of the ONNX model file
    classes : Union[Sequence[LT], Mapping[int, LT]]
        The classes that the model uses. 
    post_processor : PostProcessorType, optional
        Some post processing. Not all models need this,
        so by default PostProcessorType.IDENTITY
    factory : OnnxFactory, optional
        The model factory that is used.
        It is not needed to change this, except if you extended this 
        library. 
        By default OnnxFactory()

    Returns
    -------
    AbstractClassifier[Any, Any, Any, np.ndarray, Any, LT, np.ndarray, np.ndarray]
        An InstanceLib wrapped model
    """    
    model = factory.build_data_model(model_location, classes, post_processor, **kwargs)
    return model
