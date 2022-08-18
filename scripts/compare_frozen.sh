#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SCRIPT_DIR="${SCRIPT_DIR}/.."

SAVED_MODEL_PATH=$1
if [[ ${SAVED_MODEL_PATH} == "" ]]; then
    echo "Invalid SavedModel path: ${SAVED_MODEL_PATH}"
    exit 0
fi

OUTPUT_PATH="${SCRIPT_DIR}/output"
mkdir -p ${OUTPUT_PATH}

echo "Building CPP examples..."
cd "${SCRIPT_DIR}/cpp"
script -q -c "bash ./build.sh" /dev/null | tee "${SCRIPT_DIR}/cpp/cpp_build.log"
echo "Building CPP examples is done."

cd ${SCRIPT_DIR}
echo "Running Python Freeze"
script -q -c "python py/freeze.py --model_path=${SAVED_MODEL_PATH} --output_path=${OUTPUT_PATH}/graphs" /dev/null | tee "${OUTPUT_PATH}/freeze_py.log"
echo "Running C++ Freeze"
script -q -c "cpp/build/tf_freezer --model_dir=${SAVED_MODEL_PATH} --out_dir=${OUTPUT_PATH}/graphs" /dev/null | tee "${OUTPUT_PATH}/freeze_cpp.log"

echo "Finished freezing, comparing frozen GraphDefs..."
script -q -c "diff ${OUTPUT_PATH}/graphs/frozen_graph.pb ${OUTPUT_PATH}/graphs/frozen_graph_cpp.pb" /dev/null | tee "${OUTPUT_PATH}/diff.log"
script -q -c "python util/compare.py --first_graph_path=${OUTPUT_PATH}/graphs/frozen_graph.pb --second_graph_path=${OUTPUT_PATH}/graphs/frozen_graph_cpp.pb" /dev/null | tee "${OUTPUT_PATH}/compare.log"
