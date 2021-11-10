#%%
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from skl2onnx import convert_sklearn
from sklearn.naive_bayes import MultinomialNB
from skl2onnx.common.data_types import DoubleTensorType, FloatTensorType
from instancelib.utils.func import list_unzip

import instancelib as il
import onnxruntime as rt

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

#%% 
sess = rt.InferenceSession("output/test-model.onnx")

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

#%%
keys, vecs = test.bulk_get_vectors(test.key_list)
mat = np.vstack(vecs)
# retrieve prediction - passing in the input list (you can also pass in multiple inputs as a list of lists)
pred_onx = sess.run([label_name], {input_name: vecs})[0]

# %%
