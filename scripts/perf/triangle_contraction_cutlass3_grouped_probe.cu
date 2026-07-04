#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include <cute/tensor.hpp>

#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/kernel_hardware_info.h>

namespace {

using namespace cute;

using Element = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;
using TileShape = Shape<_128, _128, _64>;
using ClusterShape = Shape<_2, _1, _1>;
using UnderlyingProblemShape = Shape<int, int, int>;
using GroupProblemShape = cutlass::gemm::GroupProblemShape<UnderlyingProblemShape>;

// CUTLASS 3 grouped GEMM is not selected by the old GemmGrouped wrapper.  The
// grouped SM90 TMA path is selected by *pointer-backed layout tags*
// (RowMajor*/ColumnMajor*) and a GroupProblemShape.  CUTLASS then updates TMA
// descriptors per group inside one Hopper persistent scheduler.  This probe
// uses that path directly to ask whether the ragged triangle-contraction core is
// worth promoting to a larger fused triangle-multiplication kernel.
template <typename LayoutA, typename LayoutB>
struct GroupedGemmTypes {
  using FusionOperation =
      cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90AccFetch>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassTensorOp,
          TileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator,
          ElementCompute,
          void,
          cutlass::layout::RowMajor*,
          1,
          Element,
          cutlass::layout::RowMajor*,
          8,
          cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative,
          FusionOperation>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassTensorOp,
          Element,
          LayoutA,
          8,
          Element,
          LayoutB,
          8,
          ElementAccumulator,
          TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<
              static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      GroupProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue,
      void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

Element* bf16_ptr(torch::Tensor const& tensor) {
  return reinterpret_cast<Element*>(tensor.data_ptr<at::BFloat16>());
}

Element const* const_bf16_ptr(torch::Tensor const& tensor) {
  return reinterpret_cast<Element const*>(tensor.data_ptr<at::BFloat16>());
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
  TORCH_CHECK(lhs.size(2) == lhs.size(3), "probe expects square padded matrices");
  TORCH_CHECK(lhs.size(2) % 8 == 0, "TMA BF16 groups require padded N to be a multiple of 8");
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

template <typename T>
torch::Tensor device_blob_from_host(std::vector<T> const& host, torch::Device device, cudaStream_t stream) {
  auto blob = torch::empty(
      {static_cast<long>(host.size() * sizeof(T))},
      torch::TensorOptions().device(device).dtype(torch::kUInt8));
  cudaMemcpyAsync(blob.data_ptr(), host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return blob;
}

torch::Tensor device_ptrs_from_host(std::vector<Element const*> const& host, torch::Device device,
                                    cudaStream_t stream) {
  auto ptrs = torch::empty(
      {static_cast<long>(host.size())},
      torch::TensorOptions().device(device).dtype(torch::kInt64));
  cudaMemcpyAsync(ptrs.data_ptr<int64_t>(), host.data(), host.size() * sizeof(Element const*),
                  cudaMemcpyHostToDevice, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return ptrs;
}

torch::Tensor device_ptrs_from_host(std::vector<Element*> const& host, torch::Device device,
                                    cudaStream_t stream) {
  auto ptrs = torch::empty(
      {static_cast<long>(host.size())},
      torch::TensorOptions().device(device).dtype(torch::kInt64));
  cudaMemcpyAsync(ptrs.data_ptr<int64_t>(), host.data(), host.size() * sizeof(Element*),
                  cudaMemcpyHostToDevice, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return ptrs;
}

template <typename LayoutA, typename LayoutB>
torch::Tensor run_impl(torch::Tensor const& lhs, torch::Tensor const& rhs, torch::Tensor const& lengths) {
  check_inputs(lhs, rhs, lengths);

  using Types = GroupedGemmTypes<LayoutA, LayoutB>;
  using Gemm = typename Types::Gemm;
  using GemmKernel = typename Types::GemmKernel;
  using MainloopArguments = typename GemmKernel::MainloopArguments;
  using EpilogueArguments = typename GemmKernel::EpilogueArguments;
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;
  using InternalStrideA = remove_pointer_t<StrideA>;
  using InternalStrideB = remove_pointer_t<StrideB>;
  using InternalStrideD = remove_pointer_t<StrideD>;

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
  Element const* lhs_base = const_bf16_ptr(lhs);
  Element const* rhs_base = const_bf16_ptr(rhs);
  Element* out_base = bf16_ptr(out);

  std::vector<UnderlyingProblemShape> host_problem_shapes(problem_count);
  std::vector<Element const*> host_a(problem_count);
  std::vector<Element const*> host_b(problem_count);
  std::vector<Element*> host_d(problem_count);
  std::vector<InternalStrideA> host_stride_a(problem_count);
  std::vector<InternalStrideB> host_stride_b(problem_count);
  std::vector<InternalStrideD> host_stride_d(problem_count);

  for (int p = 0; p < problem_count; ++p) {
    Problem item = order[p];
    int aligned_length = ((item.length + 7) / 8) * 8;
    TORCH_CHECK(aligned_length <= n_max, "aligned problem length exceeds padded storage");
    int64_t offset = (item.feature * batch + item.batch_index) * matrix_stride;
    host_problem_shapes[p] = UnderlyingProblemShape{aligned_length, aligned_length, aligned_length};
    host_a[p] = lhs_base + offset;
    host_b[p] = rhs_base + offset;
    host_d[p] = out_base + offset;

    // The matrix stride is the padded physical N, while the problem shape is
    // the exact aligned logical length.  Padding rows/columns are zero-filled
    // by the Python harness, so over-computing to the next multiple of eight is
    // numerically safe and keeps the BF16 GMMA/TMA alignment contract explicit.
    if constexpr (cute::is_same_v<LayoutA, cutlass::layout::RowMajor*>) {
      host_stride_a[p] = InternalStrideA{n_max, _1{}, _0{}};
    } else {
      host_stride_a[p] = InternalStrideA{_1{}, n_max, _0{}};
    }
    if constexpr (cute::is_same_v<LayoutB, cutlass::layout::ColumnMajor*>) {
      host_stride_b[p] = InternalStrideB{n_max, _1{}, _0{}};
    } else {
      host_stride_b[p] = InternalStrideB{_1{}, n_max, _0{}};
    }
    host_stride_d[p] = InternalStrideD{n_max, _1{}, _0{}};
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto device = lhs.device();
  auto problem_blob = device_blob_from_host(host_problem_shapes, device, stream);
  auto stride_a_blob = device_blob_from_host(host_stride_a, device, stream);
  auto stride_b_blob = device_blob_from_host(host_stride_b, device, stream);
  auto stride_d_blob = device_blob_from_host(host_stride_d, device, stream);
  auto ptr_a = device_ptrs_from_host(host_a, device, stream);
  auto ptr_b = device_ptrs_from_host(host_b, device, stream);
  auto ptr_d = device_ptrs_from_host(host_d, device, stream);

  GroupProblemShape problem_shape{
      problem_count,
      reinterpret_cast<UnderlyingProblemShape*>(problem_blob.data_ptr<uint8_t>()),
      host_problem_shapes.data()};

  MainloopArguments mainloop_args{
      reinterpret_cast<Element const**>(ptr_a.data_ptr<int64_t>()),
      reinterpret_cast<StrideA>(stride_a_blob.data_ptr<uint8_t>()),
      reinterpret_cast<Element const**>(ptr_b.data_ptr<int64_t>()),
      reinterpret_cast<StrideB>(stride_b_blob.data_ptr<uint8_t>())};
  EpilogueArguments epilogue_args{
      {},
      nullptr,
      StrideC{},
      reinterpret_cast<Element**>(ptr_d.data_ptr<int64_t>()),
      reinterpret_cast<StrideD>(stride_d_blob.data_ptr<uint8_t>())};

  auto hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel>(
      lhs.get_device(), 0, 0, stream);
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      problem_shape,
      mainloop_args,
      epilogue_args,
      hw_info,
      {}};

  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = torch::empty(
      {static_cast<long>(workspace_size)},
      torch::TensorOptions().dtype(torch::kUInt8).device(lhs.device()));

  cutlass::Status status = gemm.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS 3 grouped can_implement failed");
  status = gemm.initialize(arguments, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS 3 grouped initialize failed");
  status = gemm.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS 3 grouped run failed");
  return out;
}

torch::Tensor run_grouped(torch::Tensor const& lhs, torch::Tensor const& rhs,
                          torch::Tensor const& lengths, bool outgoing) {
  if (outgoing) {
    return run_impl<cutlass::layout::RowMajor*, cutlass::layout::ColumnMajor*>(lhs, rhs, lengths);
  }
  return run_impl<cutlass::layout::ColumnMajor*, cutlass::layout::RowMajor*>(lhs, rhs, lengths);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &run_grouped, "SM90 CUTLASS 3 grouped triangle contraction probe");
}
