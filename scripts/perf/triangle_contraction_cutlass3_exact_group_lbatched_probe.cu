#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cstdint>

#include <cute/tensor.hpp>

#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/kernel/tile_scheduler.hpp>
#include <cutlass/kernel_hardware_info.h>

namespace protenix_cutlass3_exact_group_lbatched_probe {

using namespace cute;

using Element = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;
using TileShape = Shape<_128, _128, _64>;
using ClusterShape = Shape<_2, _1, _1>;

// A producer-owned exact-length group is stored as [L, N, N], where
// L = records_in_group * c_z.  CUTLASS sees L as GEMM's batch dimension.
using KMajorStride = Stride<int64_t, Int<1>, int64_t>;
using MnMajorStride = Stride<Int<1>, int64_t, int64_t>;

template <typename LayoutA, typename LayoutB>
struct GemmTypes {
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
          KMajorStride,
          1,
          Element,
          KMajorStride,
          8,
          cutlass::epilogue::TmaWarpSpecialized,
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
          cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

Element* bf16_ptr(torch::Tensor const& tensor) {
  return reinterpret_cast<Element*>(tensor.data_ptr<at::BFloat16>());
}

Element const* const_bf16_ptr(torch::Tensor const& tensor) {
  return reinterpret_cast<Element const*>(tensor.data_ptr<at::BFloat16>());
}

void check_inputs(torch::Tensor const& lhs, torch::Tensor const& rhs, int64_t length) {
  TORCH_CHECK(lhs.is_cuda() && rhs.is_cuda(), "lhs/rhs must be CUDA");
  TORCH_CHECK(lhs.scalar_type() == at::kBFloat16 && rhs.scalar_type() == at::kBFloat16,
              "lhs/rhs must be BF16");
  TORCH_CHECK(lhs.dim() == 3 && rhs.dim() == 3, "lhs/rhs must be [L, N, N]");
  TORCH_CHECK(lhs.is_contiguous() && rhs.is_contiguous(), "lhs/rhs must be contiguous");
  TORCH_CHECK(lhs.sizes() == rhs.sizes(), "lhs/rhs shape mismatch");
  TORCH_CHECK(lhs.size(1) == lhs.size(2), "group matrices must be square");
  TORCH_CHECK(length > 0 && length <= lhs.size(1), "invalid group length");
  TORCH_CHECK(lhs.size(1) % 8 == 0, "SM90 BF16 TMA path requires N padded to a multiple of 8");
  TORCH_CHECK(lhs.size(0) % 8 == 0, "L dimension must be tensor-core aligned");
}

template <typename LayoutA, typename LayoutB>
torch::Tensor run_group(torch::Tensor const& lhs, torch::Tensor const& rhs, int64_t length) {
  using Types = GemmTypes<LayoutA, LayoutB>;
  using Gemm = typename Types::Gemm;
  using GemmKernel = typename Types::GemmKernel;
  using MainloopArguments = typename GemmKernel::MainloopArguments;
  using EpilogueArguments = typename GemmKernel::EpilogueArguments;
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;

  int64_t l_dim = lhs.size(0);
  int64_t n = lhs.size(1);
  int aligned_length = static_cast<int>(((length + 7) / 8) * 8);
  TORCH_CHECK(aligned_length <= n, "aligned group length exceeds padded storage");

  auto out = torch::empty_like(lhs);

  auto stride_a = [&]() {
    if constexpr (cute::is_same_v<LayoutA, KMajorStride>) {
      return StrideA{n, _1{}, n * n};
    } else {
      return StrideA{_1{}, n, n * n};
    }
  }();
  auto stride_b = [&]() {
    if constexpr (cute::is_same_v<LayoutB, KMajorStride>) {
      return StrideB{n, _1{}, n * n};
    } else {
      return StrideB{_1{}, n, n * n};
    }
  }();
  StrideD stride_d{n, _1{}, n * n};

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel>(
      lhs.get_device(), 0, 0, stream);

  MainloopArguments mainloop_args{const_bf16_ptr(lhs), stride_a, const_bf16_ptr(rhs), stride_b};
  EpilogueArguments epilogue_args{{}, nullptr, StrideC{}, bf16_ptr(out), stride_d};
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {aligned_length, aligned_length, aligned_length, static_cast<int>(l_dim)},
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
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS exact-group L-batched can_implement failed: ",
              cutlassGetStatusString(status));
  status = gemm.initialize(arguments, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS exact-group L-batched initialize failed: ",
              cutlassGetStatusString(status));
  status = gemm.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS exact-group L-batched run failed: ",
              cutlassGetStatusString(status));
  return out;
}

torch::Tensor run_exact_group(torch::Tensor const& lhs, torch::Tensor const& rhs, int64_t length, bool outgoing) {
  check_inputs(lhs, rhs, length);
  torch::Tensor out;
  if (outgoing) {
    out = run_group<KMajorStride, KMajorStride>(lhs, rhs, length);
  } else {
    out = run_group<MnMajorStride, MnMajorStride>(lhs, rhs, length);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace protenix_cutlass3_exact_group_lbatched_probe

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &protenix_cutlass3_exact_group_lbatched_probe::run_exact_group,
        "SM90 CUTLASS 3 triangle contraction with one rank-4 GEMM per exact-length group");
}
