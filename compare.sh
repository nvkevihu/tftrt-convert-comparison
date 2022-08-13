#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

SAVED_MODEL_PATH=$1
if [[ ${SAVED_MODEL_PATH} == "" ]]; then
    echo "Invalid SavedModel path: ${SAVED_MODEL_PATH}"
    exit 0
fi

OUTPUT_PATH="${SCRIPT_DIR}/output"
mkdir -p ${OUTPUT_PATH}

echo "Building CPP converter..."
cd "${SCRIPT_DIR}/cpp"
script -q -c "bash ./build.sh" /dev/null | tee "${SCRIPT_DIR}/cpp/cpp_build.log"
echo "Building CPP converter is done."

cd ${SCRIPT_DIR}
script -q -c "python py/freeze.py --model_path=${SAVED_MODEL_PATH} --output_path=${OUTPUT_PATH}/graphs" /dev/null | tee "${OUTPUT_PATH}/freeze.log"
script -q -c "python py/convert.py --model_path=${SAVED_MODEL_PATH} --output_path=${OUTPUT_PATH}/graphs" /dev/null | tee "${OUTPUT_PATH}/convert_py.log"
script -q -c "${SCRIPT_DIR}/cpp/build/tftrt_converter --graph_path=${OUTPUT_PATH}/graphs/frozen_graph.pb --out_dir=${OUTPUT_PATH}/graphs" /dev/null | tee "${OUTPUT_PATH}/convert_cpp.log"

script -q -c "diff ${OUTPUT_PATH}/graphs/converted_graph_py.pb ${OUTPUT_PATH}/graphs/converted_graph_cpp.pb" /dev/null | tee "${OUTPUT_PATH}/diff.log"