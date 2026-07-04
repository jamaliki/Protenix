#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

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
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/kernel/tile_scheduler.hpp>
#include <cutlass/kernel_hardware_info.h>

namespace protenix_cutlass3_record_lbatched_probe {

using namespace cute;

using Element = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;
using TileShape = Shape<_128, _128, _64>;
using ClusterShape = Shape<_2, _1, _1>;

// Explicit affine layouts for tensors physically stored as [D, B, N, N].
// KMajor: element(i, k, d) is at i * stride_i + k + d * stride_d.
// MnMajor: element(i, k, d) is at i + k * stride_k + d * stride_d.
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
  TORCH_CHECK(lhs.size(2) % 8 == 0, "SM90 BF16 TMA path requires N padded to a multiple of 8");
  TORCH_CHECK(lhs.size(0) % 8 == 0, "feature L dimension must be BF16 tensor-core aligned");
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
void run_one_record(
    torch::Tensor const& lhs,
    torch::Tensor const& rhs,
    torch::Tensor const& out,
    int64_t record,
    int length) {
  using Types = GemmTypes<LayoutA, LayoutB>;
  using Gemm = typename Types::Gemm;
  using GemmKernel = typename Types::GemmKernel;
  using MainloopArguments = typename GemmKernel::MainloopArguments;
  using EpilogueArguments = typename GemmKernel::EpilogueArguments;
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;

  int64_t features = lhs.size(0);
  int64_t batch = lhs.size(1);
  int64_t n_max = lhs.size(2);
  int64_t matrix_stride = n_max * n_max;
  int64_t feature_stride = batch * matrix_stride;
  int aligned_length = ((length + 7) / 8) * 8;
  TORCH_CHECK(aligned_length <= n_max, "aligned record length exceeds padded storage");

  Element const* lhs_base = const_bf16_ptr(lhs) + record * matrix_stride;
  Element const* rhs_base = const_bf16_ptr(rhs) + record * matrix_stride;
  Element* out_base = bf16_ptr(out) + record * matrix_stride;

  auto stride_a = [&]() {
    if constexpr (cute::is_same_v<LayoutA, KMajorStride>) {
      return StrideA{n_max, _1{}, feature_stride};
    } else {
      return StrideA{_1{}, n_max, feature_stride};
    }
  }();
  auto stride_b = [&]() {
    if constexpr (cute::is_same_v<LayoutB, KMajorStride>) {
      return StrideB{n_max, _1{}, feature_stride};
    } else {
      return StrideB{_1{}, n_max, feature_stride};
    }
  }();
  StrideD stride_d{n_max, _1{}, feature_stride};

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel>(
      lhs.get_device(), 0, 0, stream);

  MainloopArguments mainloop_args{lhs_base, stride_a, rhs_base, stride_b};
  EpilogueArguments epilogue_args{{}, nullptr, StrideC{}, out_base, stride_d};
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {aligned_length, aligned_length, aligned_length, static_cast<int>(features)},
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
              "CUTLASS record-L-batched can_implement failed: ",
              cutlassGetStatusString(status));
  status = gemm.initialize(arguments, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS record-L-batched initialize failed: ",
              cutlassGetStatusString(status));
  status = gemm.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS record-L-batched run failed: ",
              cutlassGetStatusString(status));
}

torch::Tensor run_record_lbatched(
    torch::Tensor const& lhs,
    torch::Tensor const& rhs,
    torch::Tensor const& lengths,
    bool outgoing) {
  check_inputs(lhs, rhs, lengths);
  auto host_lengths = copy_lengths(lengths, lhs.size(2));
  auto out = torch::empty_like(lhs);
  for (int64_t b = 0; b < lhs.size(1); ++b) {
    if (outgoing) {
      run_one_record<KMajorStride, KMajorStride>(lhs, rhs, out, b, host_lengths[b]);
    } else {
      run_one_record<MnMajorStride, MnMajorStride>(lhs, rhs, out, b, host_lengths[b]);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace protenix_cutlass3_record_lbatched_probe

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &protenix_cutlass3_record_lbatched_probe::run_record_lbatched,
        "SM90 CUTLASS 3 triangle contraction with one record launch and feature L batching");
}
