#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

#include <pybind11/stl.h>

namespace {

using BFloat16 = at::BFloat16;

void check_cublas(cublasStatus_t status, char const* what) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, what, " failed with status ", int(status));
}

void check_group(torch::Tensor const& lhs, torch::Tensor const& rhs, int64_t channels) {
  TORCH_CHECK(lhs.is_cuda() && rhs.is_cuda(), "group tensors must be CUDA");
  TORCH_CHECK(lhs.scalar_type() == at::kBFloat16 && rhs.scalar_type() == at::kBFloat16,
              "group tensors must be BF16");
  TORCH_CHECK(lhs.dim() == 3 && rhs.dim() == 3, "groups must be [records * channels, N, N]");
  TORCH_CHECK(lhs.is_contiguous() && rhs.is_contiguous(), "groups must be contiguous");
  TORCH_CHECK(lhs.sizes() == rhs.sizes(), "lhs/rhs group shape mismatch");
  TORCH_CHECK(lhs.size(1) == lhs.size(2), "groups must be square");
  TORCH_CHECK(lhs.size(0) % channels == 0, "group leading dimension must be records * channels");
}

__global__ void assemble_row_major_kernel(
    BFloat16 const* __restrict__ contracted,
    BFloat16* __restrict__ update,
    int64_t group_start,
    int64_t group_rows,
    int64_t length,
    int64_t channels) {
  int64_t c = blockIdx.y * blockDim.x + threadIdx.x;
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= group_rows || c >= channels) {
    return;
  }

  int64_t pair_rows = length * length;
  int64_t record = row / pair_rows;
  int64_t pair = row - record * pair_rows;
  int64_t src = (record * channels + c) * pair_rows + pair;
  int64_t dst = (group_start + row) * channels + c;
  update[dst] = contracted[src];
}

void run_group_cublas(
    cublasHandle_t handle,
    torch::Tensor const& lhs,
    torch::Tensor const& rhs,
    torch::Tensor& contracted,
    bool outgoing) {
  int64_t length = lhs.size(1);
  int64_t batch_count = lhs.size(0);
  int64_t stride = length * length;

  auto lhs_ptr = reinterpret_cast<__nv_bfloat16 const*>(lhs.data_ptr<BFloat16>());
  auto rhs_ptr = reinterpret_cast<__nv_bfloat16 const*>(rhs.data_ptr<BFloat16>());
  auto out_ptr = reinterpret_cast<__nv_bfloat16*>(contracted.data_ptr<BFloat16>());

  // This is the same row-major-through-cuBLAS convention used by
  // triangle_contraction_cublas_probe.cu.  cuBLAS is column-major, so we swap
  // lhs/rhs and transpose flags to compute the row-major products:
  // outgoing: lhs @ rhs.T; incoming: lhs.T @ rhs.
  cublasOperation_t transa = outgoing ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transb = outgoing ? CUBLAS_OP_N : CUBLAS_OP_T;
  float alpha = 1.0f;
  float beta = 0.0f;
  auto call = [&](cublasComputeType_t compute, cublasGemmAlgo_t algo) {
    return cublasGemmStridedBatchedEx(
        handle,
        transa,
        transb,
        static_cast<int>(length),
        static_cast<int>(length),
        static_cast<int>(length),
        &alpha,
        rhs_ptr,
        CUDA_R_16BF,
        static_cast<int>(length),
        stride,
        lhs_ptr,
        CUDA_R_16BF,
        static_cast<int>(length),
        stride,
        &beta,
        out_ptr,
        CUDA_R_16BF,
        static_cast<int>(length),
        stride,
        static_cast<int>(batch_count),
        compute,
        algo);
  };

  cublasStatus_t status = call(CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (status == CUBLAS_STATUS_NOT_SUPPORTED) {
    status = call(CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  }
  check_cublas(status, "cublasGemmStridedBatchedEx exact-group contraction");
}

torch::Tensor contract_assemble(
    std::vector<torch::Tensor> const& lhs_groups,
    std::vector<torch::Tensor> const& rhs_groups,
    std::vector<int64_t> const& starts,
    int64_t rows,
    int64_t channels,
    bool outgoing) {
  TORCH_CHECK(!lhs_groups.empty(), "expected at least one group");
  TORCH_CHECK(lhs_groups.size() == rhs_groups.size(), "lhs/rhs group count mismatch");
  TORCH_CHECK(lhs_groups.size() == starts.size(), "group start count mismatch");
  TORCH_CHECK(channels > 0 && rows > 0, "invalid output shape");

  auto options = lhs_groups[0].options();
  auto update = torch::empty({rows, channels}, options);
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  check_cublas(cublasSetStream(handle, stream), "cublasSetStream");

  for (size_t index = 0; index < lhs_groups.size(); ++index) {
    auto const& lhs = lhs_groups[index];
    auto const& rhs = rhs_groups[index];
    check_group(lhs, rhs, channels);
    int64_t length = lhs.size(1);
    int64_t group_rows = (lhs.size(0) / channels) * length * length;
    TORCH_CHECK(starts[index] >= 0 && starts[index] + group_rows <= rows,
                "group compact row span exceeds output rows");

    auto contracted = torch::empty_like(lhs);
    run_group_cublas(handle, lhs, rhs, contracted, outgoing);

    dim3 block(16, 16);
    dim3 grid((group_rows + block.y - 1) / block.y, (channels + block.x - 1) / block.x);
    assemble_row_major_kernel<<<grid, block, 0, stream>>>(
        contracted.data_ptr<BFloat16>(),
        update.data_ptr<BFloat16>(),
        starts[index],
        group_rows,
        length,
        channels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return update;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("contract_assemble", &contract_assemble,
        "Exact-group cuBLAS contraction plus native row-major compact assembly");
}
