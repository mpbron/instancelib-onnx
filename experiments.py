#%%
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from skl2onnx import convert_sklearn
from sklearn.naive_bayes import MultinomialNB
from skl2onnx.common.data_types import DoubleTensorType, FloatTensorType
from instancelib.utils.func import list_unzip
from ilonnx.inference.classifier import OnnxClassifier
from instancelib.labels.encoder import DictionaryEncoder, IdentityEncoder

import instancelib as il
import onnxruntime as rt

from ilonnx.inference.factory import OnnxFactory

#%%
# Train and fit the model
env = il.read_excel_dataset("datasets/testdataset.xlsx", ["fulltext"], ["label"])
vect = il.TextInstanceVectorizer(il.SklearnVectorizer(TfidfVectorizer(max_features=1000)))
il.vectorize(vect, env)
train, test = env.train_test_split(env.dataset, 0.70)
model = il.SkLearnVectorClassifier.build(MultinomialNB(), env)
model.fit_provider(train, env.labels)

#%%
model_onnx = convert_sklearn(model.innermodel,
                             initial_types=[("input", FloatTensorType([None, 1000]))]
                            )

# Write to disk
with open("output/test-model.onnx", "wb") as f:
    f.write(model_onnx.SerializeToString())

# %% 
factory = OnnxFactory()
read_model = factory.build_vector_model(
    "output/test-model.onnx", 
    classes=["Bedrijfsnieuws", "Games", "Smartphones"])
#%
#%%
performance = il.classifier_performance_mc(read_model, test, env.labels)

#%%
read_model.predict_proba(test)
# %%
