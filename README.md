# instancelib-onnx
ONNX extension for instancelib

```python
import instancelib as il
import instancelib as ilonnx

# Specify the model location and the label translation 
model = ilonnx.build_data_model("example_models/data-model.onnx", 
                                {0: "Bedrijfsnieuws", 1: "Games", 2: "Smartphones"})
```

Then you can use the normal instancelib functionality to interact with the model.

```python
# Load a dataset with instancelib
env = il.read_excel_dataset("datasets/testdataset.xlsx", ["fulltext"], ["label"])

# Assess the performance like any other instancelib model
performance = il.classifier_performance(read_data_model, env.dataset, env.labels)
performance.confusion_matrix
```