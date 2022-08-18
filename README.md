# TF-TRT Conversion Comparison

The purpose of this repo is to compare TF to TensorRT conversion using the Python vs. C++ APIs. This is done by:
1. Loading a SavedModel in Python, freezing it, and serializing the frozen GraphDef to disk
2. Loading the GraphDef in Python, converting it, and serializing the converted GraphDef to disk
3. Loading the GraphDef in C++, converting it, and serializing the converted GraphDef to disk
4. Comparing the results of (2) and (3) to ensure they are identical

Some changes are needed to the current C++ API to do this:
- Ensure that optimizations run are identical between C++ and Python
- Change `ConvertAndBuild` to only convert and return the converted GraphDef

Additionally, for step (2), the current script (in `py/convert.py`) will load the SavedModel, rather than the frozen GraphDef. This may create issues for the comparison if the Python conversion does not freeze the model in the same way as in the freeze script (in `py/freeze.py`).

## TF Patch

Update the C++ conversion API to better match the Python one by applying the included patch file. In the 22.08 container:

```
cd /opt/tensorflow/tensorflow-source
patch -p1 < /workspace/tftrt-convert-comparison/patch/tftrt-convert.patch
patch -p1 < /workspace/tftrt-convert-comparison/patch/trt-build-verbose.patch
```

## Dependencies

Visualizing built engines relies on TREx, found [here](https://github.com/NVIDIA/TensorRT/tree/b55c4710ce01f076c26710a48879fcb2661be4a9/tools/experimental/trt-engine-explorer). Install this with pip:

```
cd [TRex directory] && pip install .
```

Graphviz will also be needed:

```
apt-get update
apt-get install graphviz
```

## Running

To do this comparison for a particular SavedModel, for example:

```
bash scripts/compare_partial.sh /models/huggingface/transformers/bert_base_uncased/pb_model/
```

The script performs the above steps in order. Note that this only works OOTB for some transformer models, as input shapes are currently hardcoded for engine building.

Additional scripts are provided for comparing the results of freezing in C++ vs. Python (in `scripts/compare_frozen.sh`), and for comparing C++ conversion from a SavedModel, rather than a frozen GraphDef (in `scripts/compare_e2e.sh`).
