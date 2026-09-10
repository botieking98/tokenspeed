#include <torch/library.h>
#include <torch/torch.h>

#include "aclnn_torch_adapter/op_api_common.h"

namespace tokenspeed_kernel {

void npu_scatter_nd_update_v2(
    at::Tensor& var,
    const at::Tensor& indices,
    const at::Tensor& update) {
    TORCH_CHECK(
        indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong,
        "scatter indices must be int32 or int64");
    at::IntArrayRef var_stride = var.strides();
    at::Tensor indices_int64 =
        indices.scalar_type() == at::kLong ? indices : indices.to(at::kLong);
    EXEC_NPU_CMD(aclnnScatterNdUpdateV2, var, indices_int64, update, var_stride);
}

}  // namespace tokenspeed_kernel

TORCH_LIBRARY(tokenspeed_ascend, m) {
    m.def("npu_scatter_nd_update_v2(Tensor(a!) var, Tensor indices, Tensor update) -> ()");
    m.impl(
        "npu_scatter_nd_update_v2",
        torch::kPrivateUse1,
        &tokenspeed_kernel::npu_scatter_nd_update_v2);
}
