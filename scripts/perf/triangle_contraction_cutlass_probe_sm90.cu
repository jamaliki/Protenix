#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

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

namespace {

using namespace cute;

using Element = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;

// This is intentionally only the triangular contraction core:
//
//   out[d,b,i,j] = sum_k lhs[d,b,i,k] * rhs[d,b,j,k]
//
// The current CUEQ triangle-multiplication wrapper forms lhs/rhs with fused
// LayerNorm and gated dual-GEMM kernels, then performs this contraction as a
// stack of small square GEMMs.  Testing this boundary first tells us whether our
// native schedule can match the vendor-quality dense contraction before we fuse
// the surrounding projection and output-gate stages.
using TileShape = Shape<_64, _64, _64>;
using ClusterShape = Shape<_1, _1, _1>;

using FusionOperation =
    cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90AccFetch>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    TileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementCompute,
    void,
    cutlass::layout::RowMajor,
    1,
    Element,
    cutlass::layout::RowMajor,
    8,
    cutlass::epilogue::TmaWarpSpecialized,
    FusionOperation>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    Element,
    cutlass::layout::RowMajor,
    8,
    Element,
    cutlass::layout::ColumnMajor,
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
using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;
using MainloopArguments = typename Gemm::GemmKernel::MainloopArguments;
using EpilogueArguments = typename Gemm::GemmKernel::EpilogueArguments;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

Element const* bf16_ptr(torch::Tensor const& tensor) {
  return reinterpret_cast<Element const*>(tensor.data_ptr<at::BFloat16>());
}

Element* mutable_bf16_ptr(torch::Tensor const& tensor) {
  return reinterpret_cast<Element*>(tensor.data_ptr<at::BFloat16>());
}

void check_inputs(torch::Tensor const& lhs, torch::Tensor const& rhs) {
  TORCH_CHECK(lhs.is_cuda() && rhs.is_cuda(), "lhs/rhs must be CUDA tensors");
  TORCH_CHECK(lhs.scalar_type() == at::kBFloat16, "lhs must be BF16");
  TORCH_CHECK(rhs.scalar_type() == at::kBFloat16, "rhs must be BF16");
  TORCH_CHECK(lhs.dim() == 4 && rhs.dim() == 4, "lhs/rhs must be [D, B, M, K]");
  TORCH_CHECK(lhs.is_contiguous() && rhs.is_contiguous(), "lhs/rhs must be contiguous");
  TORCH_CHECK(lhs.size(0) == rhs.size(0), "feature dimension mismatch");
  TORCH_CHECK(lhs.size(1) == rhs.size(1), "batch dimension mismatch");
  TORCH_CHECK(lhs.size(3) == rhs.size(3), "contraction dimension mismatch");
  TORCH_CHECK(lhs.size(2) <= 4096 && rhs.size(2) <= 4096 && lhs.size(3) <= 4096,
              "probe expects ordinary protein token dimensions");
}

torch::Tensor run_contract(torch::Tensor const& lhs, torch::Tensor const& rhs) {
  check_inputs(lhs, rhs);

  int64_t features = lhs.size(0);
  int64_t batch = lhs.size(1);
  int64_t m = lhs.size(2);
  int64_t n = rhs.size(2);
  int64_t k = lhs.size(3);
  int64_t gemm_batch = features * batch;

  auto out = torch::empty({features, batch, m, n}, lhs.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel>(
      lhs.get_device(), 0, 0, stream);

  ProblemShapeType problem_shape{
      static_cast<int>(m),
      static_cast<int>(n),
      static_cast<int>(k),
      static_cast<int>(gemm_batch)};

  // lhs is row-major [M, K].  rhs is stored row-major [N, K], which is exactly
  // column-major storage for the logical [K, N] operand needed by GEMM.
  MainloopArguments mainloop_args{
      bf16_ptr(lhs),
      StrideA{k, _1{}, m * k},
      bf16_ptr(rhs),
      StrideB{k, _1{}, n * k}};
  EpilogueArguments epilogue_args{
      {},
      nullptr,
      StrideC{},
      mutable_bf16_ptr(out),
      StrideD{n, _1{}, m * n}};

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
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
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS can_implement failed");
  status = gemm.initialize(arguments, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS initialize failed");
  status = gemm.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS run failed");
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &run_contract, "SM90 CUTLASS triangle contraction probe");
}
