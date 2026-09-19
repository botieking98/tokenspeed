#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ASCEND_DIR="${ROOT_DIR}/thirdparty/vllm-ascend"
VLLM_ASCEND_CSRC="${VLLM_ASCEND_DIR}/csrc"
SCATTER_OVERRIDES_DIR="${ROOT_DIR}/csrc/cann_ops/moe/scatter_nd_update_v2"
VLLM_SCATTER_DIR="${VLLM_ASCEND_CSRC}/moe/scatter_nd_update_v2"
BUILD_DIR="${ROOT_DIR}/build/ascend"
SOC_VERSION="${1:?usage: build_cann_ops.sh <soc-version> <jobs> <python>}"
JOBS="${2:?usage: build_cann_ops.sh <soc-version> <jobs> <python>}"
PYTHON_BIN="${3:?usage: build_cann_ops.sh <soc-version> <jobs> <python>}"
CANN_HOME="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable was not found: ${PYTHON_BIN}" >&2
    exit 1
fi
PYTHON_BIN="$("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
export CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}"

EXTENSION_SOC_VERSION="${SOC_VERSION}"
if [[ "${SOC_VERSION}" == "ascend910_93" ]]; then
    EXTENSION_SOC_VERSION="$(
        "${PYTHON_BIN}" -c 'import torch; import torch_npu; print(torch.npu.get_device_name(0).lower())'
    )"
fi

VLLM_CANN_OPS=(
    compressor
    compressor_metadata
    dispatch_ffn_combine
    grouped_matmul_swiglu_quant_weight_nz_tensor_list
    hc_post
    hc_pre
    inplace_partial_rotary_mul
    moe_gating_top_k_hash
    rms_norm_dynamic_quant
    sparse_attn_sharedkv
    sparse_attn_sharedkv_metadata
    vllm_quant_lightning_indexer
    vllm_quant_lightning_indexer_metadata
)

CANN_OPS=(
    "${VLLM_CANN_OPS[@]}"
    scatter_nd_update_v2
)

function join_by_comma() {
    local separator=','
    local result=''
    local item
    for item in "$@"; do
        if [[ -n "${result}" ]]; then
            result="${result}${separator}${item}"
        else
            result="${item}"
        fi
    done
    printf '%s' "${result}"
}

function sync_symlink() {
    local target="$1"
    local destination="$2"

    if [[ -L "${destination}" && "$(readlink "${destination}")" == "${target}" ]]; then
        return
    fi
    if [[ -d "${destination}" && ! -L "${destination}" ]]; then
        echo "Refusing to replace directory with symlink: ${destination}" >&2
        exit 1
    fi
    ln -sfn "${target}" "${destination}"
}

function install_cann_run_package() {
    local source_dir="$1"
    local destination="$2"
    local installer

    installer="$(
        find "${source_dir}/build" -maxdepth 1 -type f -name 'cann-ops-transformer*.run' -print -quit
    )"
    if [[ -z "${installer}" ]]; then
        echo "CANN run package was not found under ${source_dir}/build" >&2
        exit 1
    fi

    rm -rf "${destination}"
    mkdir -p "${destination}"
    chmod +x "${installer}"
    "${installer}" --install-path="${destination}"
}

function prepare_vllm_ascend_sources() {
    if [[ ! -f "${VLLM_ASCEND_CSRC}/CMakeLists.txt" ]]; then
        git -C "${ROOT_DIR}" submodule update --init --recursive
    fi
    if [[ ! -f "${VLLM_ASCEND_CSRC}/CMakeLists.txt" ]]; then
        echo "vllm-ascend source submodule is missing" >&2
        exit 1
    fi
    if [[ ! -d "${VLLM_ASCEND_CSRC}/third_party/catlass/include" ]]; then
        git -C "${VLLM_ASCEND_DIR}" submodule update --init csrc/third_party/catlass
    fi
    if [[ ! -d "${VLLM_ASCEND_CSRC}/third_party/catlass/include" ]]; then
        echo "vllm-ascend catlass dependency is missing" >&2
        exit 1
    fi
}

function build_tokenspeed_extension() {
    local build_root="${BUILD_DIR}/tokenspeed-extension"
    local torch_cmake_prefix

    torch_cmake_prefix="$(
        "${PYTHON_BIN}" -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parent / "share" / "cmake")'
    )"
    rm -rf "${build_root}"
    cmake \
        -S "${ROOT_DIR}/csrc" \
        -B "${build_root}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="${torch_cmake_prefix}" \
        -DPYTHON_EXECUTABLE="${PYTHON_BIN}" \
        -DASCEND_HOME_PATH="${CANN_HOME}"
    cmake --build "${build_root}" --parallel "${JOBS}"
}

function build_cann_ops() {
    local ops
    local staging_dir="${BUILD_DIR}/cann-source"
    local entry_name
    local scatter_staging_dir

    ops="$(join_by_comma "${CANN_OPS[@]}")"

    mkdir -p "${staging_dir}/moe"
    rm -f \
        "${staging_dir}/CMakeLists.txt" \
        "${staging_dir}/build.sh" \
        "${staging_dir}/moe/CMakeLists.txt"
    cp -a "${VLLM_ASCEND_CSRC}/CMakeLists.txt" "${staging_dir}/CMakeLists.txt"
    cp -a "${VLLM_ASCEND_CSRC}/build.sh" "${staging_dir}/build.sh"
    cp -a "${VLLM_ASCEND_CSRC}/moe/CMakeLists.txt" "${staging_dir}/moe/CMakeLists.txt"
    scatter_staging_dir="${staging_dir}/moe/scatter_nd_update_v2"
    rm -rf "${scatter_staging_dir}"
    cp -a "${VLLM_SCATTER_DIR}" "${scatter_staging_dir}"
    cp -a \
        "${SCATTER_OVERRIDES_DIR}/op_host/scatter_nd_update_v2_tiling.cpp" \
        "${scatter_staging_dir}/op_host/scatter_nd_update_v2_tiling.cpp"
    cp -a \
        "${SCATTER_OVERRIDES_DIR}/op_kernel/scatter_nd_update_large_index.h" \
        "${scatter_staging_dir}/op_kernel/scatter_nd_update_large_index.h"

    for entry_name in \
        attention \
        common \
        cmake \
        ffn \
        gmm \
        mc2 \
        moe \
        posembedding \
        scripts \
        third_party \
        utils \
        version.info; do
        if [[ "${entry_name}" == "moe" ]]; then
            for moe_entry in "${VLLM_ASCEND_CSRC}/moe"/*; do
                entry_name="$(basename "${moe_entry}")"
                if [[ "${entry_name}" != "CMakeLists.txt" && "${entry_name}" != "scatter_nd_update_v2" ]]; then
                    sync_symlink "${moe_entry}" "${staging_dir}/moe/${entry_name}"
                fi
            done
        elif [[ -e "${VLLM_ASCEND_CSRC}/${entry_name}" ]]; then
            sync_symlink "${VLLM_ASCEND_CSRC}/${entry_name}" "${staging_dir}/${entry_name}"
        fi
    done

    (
        cd "${staging_dir}"
        bash ./build.sh \
            --pkg \
            --ops="${ops}" \
            --soc="${SOC_VERSION}" \
            --vendor_name=custom \
            -j"${JOBS}"
    )
    install_cann_run_package \
        "${staging_dir}" \
        "${BUILD_DIR}/cann-ops"
}

function build_vllm_ascend_extension() {
    local build_root="${BUILD_DIR}/vllm-ascend-extension-build"
    local install_root="${BUILD_DIR}/vllm-ascend-extension"
    local pybind_cmake_dir
    local python_include_dir
    local torch_npu_path

    pybind_cmake_dir="$("${PYTHON_BIN}" -m pybind11 --cmakedir)"
    python_include_dir="$(
        "${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_paths()["include"])'
    )"
    torch_npu_path="$(
        "${PYTHON_BIN}" -c 'from pathlib import Path; import torch_npu; print(Path(torch_npu.__file__).resolve().parent)'
    )"

    rm -rf "${build_root}" "${install_root}"
    cmake \
        -S "${VLLM_ASCEND_DIR}" \
        -B "${build_root}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${install_root}" \
        -DCMAKE_PREFIX_PATH="${pybind_cmake_dir}" \
        -DPYTHON_EXECUTABLE="${PYTHON_BIN}" \
        -DPYTHON_INCLUDE_PATH="${python_include_dir}" \
        -DASCEND_HOME_PATH="${CANN_HOME}" \
        -DSOC_VERSION="${EXTENSION_SOC_VERSION}" \
        -DTORCH_NPU_PATH="${torch_npu_path}" \
        -DFETCHCONTENT_BASE_DIR="${BUILD_DIR}/fetchcontent"
    cmake --build "${build_root}" --target vllm_ascend_C --parallel "${JOBS}"
    cmake --install "${build_root}"
}

function collect_cann_vendors() {
    local vendor_root="${BUILD_DIR}/cann_ops/vendors"
    rm -rf "${BUILD_DIR}/cann_ops"
    mkdir -p "${vendor_root}"
    cp -a "${BUILD_DIR}/cann-ops/vendors/." "${vendor_root}/"
}

prepare_vllm_ascend_sources
export ASCEND_HOME_PATH="${CANN_HOME}"
build_tokenspeed_extension
build_cann_ops
build_vllm_ascend_extension
collect_cann_vendors

echo "TokenSpeed Ascend build artifacts are ready under ${BUILD_DIR}"
