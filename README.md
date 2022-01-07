# instancelib-onnx
ONNX extension for instancelib.

## Installation
You can install this package as follows:

```bash
pip install instancelib-onnx
```

Or by cloning this repo and issuing:
```bash
python setup.py
```

You will need at least Python 3.8 to use this library.

## Usage

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
performance = il.classifier_performance(model, env.dataset, env.labels)
performance.confusion_matrix
```