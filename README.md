# TF-TRT Conversion Comparison

The purpose of this repo is to compare TF to TensorRT conversion using the Python vs. C++ APIs. This is done by:
1. Loading a SavedModel in Python, freezing it, and serializing the frozen GraphDef to disk
2. Loading the GraphDef in Python, converting it, and serializing the converted GraphDef to disk
3. Loading the GraphDef in C++, converting it, and serializing the converted GraphDef to disk
4. Comparing the results of (2) and (3) to ensure they are identical

Some changes are needed to the current C++ API to do this:
- Ensure that optimizations run are identical between C++ and Python
- Change `ConvertAndBuild` to only convert and return the converted GraphDef

## Running

To do this comparison for a particular SavedModel:

```
bash compare.sh /path/to/saved/model/dir
```

The script performs the above steps in order.
