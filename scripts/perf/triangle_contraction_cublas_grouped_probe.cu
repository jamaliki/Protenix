#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace {

using BFloat16 = at::BFloat16;

void check_cublas(cublasStatus_t status, char const* what) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, what, " failed with status ", int(status));
}

void check_inputs(
    torch::Tensor const& lhs,
    torch::Tensor const& rhs,
    torch::Tensor const& lengths) {
  TORCH_CHECK(lhs.is_cuda() && rhs.is_cuda() && lengths.is_cuda(), "all inputs must be CUDA tensors");
  TORCH_CHECK(lhs.scalar_type() == at::kBFloat16 && rhs.scalar_type() == at::kBFloat16,
              "lhs/rhs must be BF16");
  TORCH_CHECK(lengths.scalar_type() == at::kInt, "lengths must be int32");
  TORCH_CHECK(lhs.dim() == 4 && rhs.dim() == 4, "lhs/rhs must be [D, B, Nmax, Nmax]");
  TORCH_CHECK(lengths.dim() == 1, "lengths must be [B]");
  TORCH_CHECK(lhs.is_contiguous() && rhs.is_contiguous(), "lhs/rhs must be contiguous");
  TORCH_CHECK(lhs.sizes() == rhs.sizes(), "lhs/rhs shape mismatch");
  TORCH_CHECK(lengths.size(0) == lhs.size(1), "lengths must match batch dimension");
  TORCH_CHECK(lhs.size(2) == lhs.size(3), "probe expects square padded pair tensors");
}

std::vector<int> copy_lengths(torch::Tensor const& lengths, int64_t n_max) {
  auto host = lengths.to(torch::kCPU);
  auto ptr = host.data_ptr<int32_t>();
  std::vector<int> out(host.numel());
  for (int64_t i = 0; i < host.numel(); ++i) {
    int value = static_cast<int>(ptr[i]);
    TORCH_CHECK(value > 0 && value <= n_max, "invalid sequence length");
    out[i] = value;
  }
  return out;
}

void run_one_record(
    cublasHandle_t handle,
    __nv_bfloat16 const* lhs_ptr,
    __nv_bfloat16 const* rhs_ptr,
    __nv_bfloat16* out_ptr,
    int64_t batch,
    int64_t n_max,
    int64_t b,
    int length,
    int64_t features,
    bool outgoing) {
  float alpha = 1.0f;
  float beta = 0.0f;
  int64_t matrix_stride = n_max * n_max;
  int64_t feature_stride = batch * matrix_stride;
  int64_t offset = b * matrix_stride;
  cublasOperation_t transa = outgoing ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transb = outgoing ? CUBLAS_OP_N : CUBLAS_OP_T;

  auto call = [&](cublasComputeType_t compute, cublasGemmAlgo_t algo) {
    return cublasGemmStridedBatchedEx(
        handle,
        transa,
        transb,
        length,
        length,
        length,
        &alpha,
        rhs_ptr + offset,
        CUDA_R_16BF,
        static_cast<int>(n_max),
        feature_stride,
        lhs_ptr + offset,
        CUDA_R_16BF,
        static_cast<int>(n_max),
        feature_stride,
        &beta,
        out_ptr + offset,
        CUDA_R_16BF,
        static_cast<int>(n_max),
        feature_stride,
        static_cast<int>(features),
        compute,
        algo);
  };

  cublasStatus_t status = call(CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (status == CUBLAS_STATUS_NOT_SUPPORTED) {
    status = call(CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  }
  check_cublas(status, "cublasGemmStridedBatchedEx");
}

torch::Tensor run_record_strided(
    torch::Tensor const& lhs,
    torch::Tensor const& rhs,
    torch::Tensor const& lengths,
    bool outgoing) {
  check_inputs(lhs, rhs, lengths);

  int64_t features = lhs.size(0);
  int64_t batch = lhs.size(1);
  int64_t n_max = lhs.size(2);
  std::vector<int> host_lengths = copy_lengths(lengths, n_max);

  auto out = torch::zeros_like(lhs);
  auto lhs_ptr = reinterpret_cast<__nv_bfloat16 const*>(lhs.data_ptr<BFloat16>());
  auto rhs_ptr = reinterpret_cast<__nv_bfloat16 const*>(rhs.data_ptr<BFloat16>());
  auto out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<BFloat16>());
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  auto stream = at::cuda::getCurrentCUDAStream();
  check_cublas(cublasSetStream(handle, stream), "cublasSetStream");
  for (int64_t b = 0; b < batch; ++b) {
    // This is deliberately a probe, not the final kernel: it keeps the padded
    // storage and skips invalid k/i/j work, but still pays one cuBLAS launch per
    // sequence record.  A real CuTe kernel should preserve the exact-length work
    // reduction without this launch fan-out.
    run_one_record(
        handle, lhs_ptr, rhs_ptr, out_ptr, batch, n_max, b, host_lengths[b], features, outgoing);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &run_record_strided, "record-strided cuBLAS triangle contraction probe");
}
