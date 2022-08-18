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

echo "Building CPP converter..."
cd "${SCRIPT_DIR}/cpp"
script -q -c "bash ./build.sh" /dev/null | tee "${SCRIPT_DIR}/cpp/cpp_build.log"
echo "Building CPP converter is done."

cd ${SCRIPT_DIR}
echo "Running Python Conversion"
script -q -c "TF_TRT_EXPORT_TRT_ENGINES_PATH=${OUTPUT_PATH}/trt_engines/py python py/convert.py --model_path=${SAVED_MODEL_PATH} --output_path=${OUTPUT_PATH}/graphs" /dev/null | tee "${OUTPUT_PATH}/convert_py.log"
echo "Running C++ Conversion"
script -q -c "TF_CPP_VMODULE=trt_convert_api=4 ${SCRIPT_DIR}/cpp/build/tftrt_e2e_converter --model_dir=${SAVED_MODEL_PATH} --out_dir=${OUTPUT_PATH}/graphs --engine_out_dir=${OUTPUT_PATH}/trt_engines/cpp" /dev/null | tee "${OUTPUT_PATH}/convert_cpp.log"

echo "Copying TRTEngineOp Dims from Py to C++"
cp ${OUTPUT_PATH}/trt_engines/py/dims-* ${OUTPUT_PATH}/trt_engines/cpp

echo "Running trtexec and TREx"
for engine_file in ${OUTPUT_PATH}/trt_engines/*/TRTEngineOp*; do
    ENGINE_NAME=$(basename -- ${engine_file})
    ENGINE_DIR=$(dirname -- ${engine_file})
    GRAPH_FILE="graph-${ENGINE_NAME}.json"

    cd "${ENGINE_DIR}"
    script -q -c "trtexec --loadEngine=${ENGINE_NAME} --exportLayerInfo=${GRAPH_FILE} --profilingVerbosity=detailed" /dev/null | tee "trtexec-${ENGINE_NAME}.log"
    python ${SCRIPT_DIR}/util/plot.py --plan_path ${GRAPH_FILE}
done
cd ${SCRIPT_DIR}

echo "Finished converting, comparing converted GraphDefs..."
script -q -c "diff ${OUTPUT_PATH}/graphs/converted_graph_py.pb ${OUTPUT_PATH}/graphs/converted_graph_cpp.pb" /dev/null | tee "${OUTPUT_PATH}/diff.log"
script -q -c "python util/compare.py --first_graph_path=${OUTPUT_PATH}/graphs/converted_graph_py.pb --second_graph_path=${OUTPUT_PATH}/graphs/converted_graph_cpp.pb" /dev/null | tee "${OUTPUT_PATH}/compare.log"
