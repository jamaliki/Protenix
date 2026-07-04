#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/numeric_types.h>

namespace {

using Element = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;
using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 64>;
using WarpShape = cutlass::gemm::GemmShape<32, 64, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    Element, 8, ElementAccumulator, ElementCompute>;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

Element* bf16_ptr(torch::Tensor const& tensor) {
  return reinterpret_cast<Element*>(tensor.data_ptr<at::BFloat16>());
}

void check_inputs(torch::Tensor const& lhs, torch::Tensor const& rhs, torch::Tensor const& lengths) {
  TORCH_CHECK(lhs.is_cuda() && rhs.is_cuda() && lengths.is_cuda(), "all inputs must be CUDA");
  TORCH_CHECK(lhs.scalar_type() == at::kBFloat16 && rhs.scalar_type() == at::kBFloat16,
              "lhs/rhs must be BF16");
  TORCH_CHECK(lengths.scalar_type() == at::kInt, "lengths must be int32");
  TORCH_CHECK(lhs.dim() == 4 && rhs.dim() == 4, "lhs/rhs must be [D, B, N, N]");
  TORCH_CHECK(lhs.is_contiguous() && rhs.is_contiguous(), "lhs/rhs must be contiguous");
  TORCH_CHECK(lhs.sizes() == rhs.sizes(), "lhs/rhs shape mismatch");
  TORCH_CHECK(lengths.size(0) == lhs.size(1), "lengths must match batch dimension");
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

template <typename LayoutA, typename LayoutB>
torch::Tensor run_impl(torch::Tensor const& lhs, torch::Tensor const& rhs, torch::Tensor const& lengths) {
  check_inputs(lhs, rhs, lengths);
  int64_t features = lhs.size(0);
  int64_t batch = lhs.size(1);
  int64_t n_max = lhs.size(2);
  int64_t matrix_stride = n_max * n_max;
  std::vector<int> host_lengths = copy_lengths(lengths, n_max);
  int problem_count = static_cast<int>(features * batch);

  struct Problem {
    int length;
    int64_t feature;
    int64_t batch_index;
  };
  std::vector<Problem> order;
  order.reserve(problem_count);
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t d = 0; d < features; ++d) {
      order.push_back({host_lengths[b], d, b});
    }
  }
  std::stable_sort(order.begin(), order.end(), [](Problem const& a, Problem const& b) {
    return a.length > b.length;
  });

  auto out = torch::zeros_like(lhs);
  Element* lhs_base = bf16_ptr(lhs);
  Element* rhs_base = bf16_ptr(rhs);
  Element* out_base = bf16_ptr(out);

  std::vector<cutlass::gemm::GemmCoord> host_problem_sizes(problem_count);
  std::vector<Element*> host_a(problem_count);
  std::vector<Element*> host_b(problem_count);
  std::vector<Element*> host_c(problem_count);
  std::vector<Element*> host_d(problem_count);
  std::vector<int64_t> host_lda(problem_count, n_max);
  std::vector<int64_t> host_ldb(problem_count, n_max);
  std::vector<int64_t> host_ldc(problem_count, n_max);
  std::vector<int64_t> host_ldd(problem_count, n_max);

  for (int p = 0; p < problem_count; ++p) {
    auto item = order[p];
    int64_t offset = (item.feature * batch + item.batch_index) * matrix_stride;
    host_problem_sizes[p] = cutlass::gemm::GemmCoord(item.length, item.length, item.length);
    host_a[p] = lhs_base + offset;
    host_b[p] = rhs_base + offset;
    host_c[p] = out_base + offset;
    host_d[p] = out_base + offset;
  }

  auto opts_i64 = torch::TensorOptions().device(lhs.device()).dtype(torch::kInt64);
  auto opts_i32 = torch::TensorOptions().device(lhs.device()).dtype(torch::kInt32);
  auto ptr_a = torch::empty({problem_count}, opts_i64);
  auto ptr_b = torch::empty({problem_count}, opts_i64);
  auto ptr_c = torch::empty({problem_count}, opts_i64);
  auto ptr_d = torch::empty({problem_count}, opts_i64);
  auto lda = torch::empty({problem_count}, opts_i64);
  auto ldb = torch::empty({problem_count}, opts_i64);
  auto ldc = torch::empty({problem_count}, opts_i64);
  auto ldd = torch::empty({problem_count}, opts_i64);
  auto problem_sizes = torch::empty({problem_count, 3}, opts_i32);

  auto stream = at::cuda::getCurrentCUDAStream();
  cudaMemcpyAsync(ptr_a.data_ptr(), host_a.data(), problem_count * sizeof(Element*), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(ptr_b.data_ptr(), host_b.data(), problem_count * sizeof(Element*), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(ptr_c.data_ptr(), host_c.data(), problem_count * sizeof(Element*), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(ptr_d.data_ptr(), host_d.data(), problem_count * sizeof(Element*), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(lda.data_ptr(), host_lda.data(), problem_count * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(ldb.data_ptr(), host_ldb.data(), problem_count * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(ldc.data_ptr(), host_ldc.data(), problem_count * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(ldd.data_ptr(), host_ldd.data(), problem_count * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(problem_sizes.data_ptr(), host_problem_sizes.data(),
                  problem_count * sizeof(cutlass::gemm::GemmCoord), cudaMemcpyHostToDevice, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  using Kernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
      Element, LayoutA, cutlass::ComplexTransform::kNone, 8,
      Element, LayoutB, cutlass::ComplexTransform::kNone, 8,
      Element, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOp, Swizzle, 3,
      cutlass::gemm::kernel::GroupScheduleMode::kHostPrecompute>::GemmKernel;
  using Gemm = cutlass::gemm::device::GemmGrouped<Kernel>;

  typename Gemm::Arguments args{
      reinterpret_cast<cutlass::gemm::GemmCoord*>(problem_sizes.data_ptr<int32_t>()),
      problem_count,
      Gemm::sufficient(host_problem_sizes.data(), problem_count),
      typename EpilogueOp::Params{ElementCompute(1), ElementCompute(0)},
      reinterpret_cast<Element**>(ptr_a.data_ptr<int64_t>()),
      reinterpret_cast<Element**>(ptr_b.data_ptr<int64_t>()),
      reinterpret_cast<Element**>(ptr_c.data_ptr<int64_t>()),
      reinterpret_cast<Element**>(ptr_d.data_ptr<int64_t>()),
      reinterpret_cast<int64_t*>(lda.data_ptr<int64_t>()),
      reinterpret_cast<int64_t*>(ldb.data_ptr<int64_t>()),
      reinterpret_cast<int64_t*>(ldc.data_ptr<int64_t>()),
      reinterpret_cast<int64_t*>(ldd.data_ptr<int64_t>()),
      host_problem_sizes.data()};

  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(args);
  auto workspace = torch::empty(
      {static_cast<long>(workspace_size)},
      torch::TensorOptions().dtype(torch::kUInt8).device(lhs.device()));
  cutlass::Status status = gemm.can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS grouped can_implement failed");
  status = gemm.initialize(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS grouped initialize failed");
  status = gemm.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS grouped run failed");
  return out;
}

torch::Tensor run_grouped(torch::Tensor const& lhs, torch::Tensor const& rhs,
                          torch::Tensor const& lengths, bool outgoing) {
  if (outgoing) {
    return run_impl<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>(lhs, rhs, lengths);
  }
  return run_impl<cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>(lhs, rhs, lengths);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &run_grouped, "CUTLASS grouped triangle contraction probe");
}
