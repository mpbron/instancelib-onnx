#%%
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from skl2onnx import convert_sklearn
from sklearn.naive_bayes import MultinomialNB
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from sklearn.pipeline import Pipeline

import ilonnx
import instancelib as il

from sklearn.pipeline import Pipeline # type: ignore
from sklearn.naive_bayes import MultinomialNB # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer # type: ignore

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
pipeline = Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', MultinomialNB()),
     ])
data_model = il.SkLearnDataClassifier.build(pipeline, env)
data_model.fit_provider(train, env.labels)
#%%
model_onnx = convert_sklearn(model.innermodel,
                             initial_types=[("input", FloatTensorType([None, 1000]))]
                            )

# Write to disk
with open("output/test-model.onnx", "wb") as f:
    f.write(model_onnx.SerializeToString())

#%%
model_onnx = convert_sklearn(data_model.innermodel, "pipe",
                             initial_types=[("input", StringTensorType([None]))]
                            )

# Write to disk
with open("output/data-model.onnx", "wb") as f:
    f.write(model_onnx.SerializeToString())


# %% 
read_model = ilonnx.build_vector_model(
    "output/test-model.onnx", # The model location
    {0: "Bedrijfsnieuws", 1: "Games", 2: "Smartphones"}) 

#%% 
read_data_model = ilonnx.build_data_model(
    "output/data-model.onnx",
    {0: "Bedrijfsnieuws", 1: "Games", 2: "Smartphones"}
)
#%
#%%
performance = il.classifier_performance(read_data_model, test, env.labels)
performance.confusion_matrix
#%%
read_data_model.predict(test)
# %%
