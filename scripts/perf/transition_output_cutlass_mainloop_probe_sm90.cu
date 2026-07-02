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
static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

using TileShape = Shape<_128, _128, _64>;
using ClusterShape = Shape<_2, _1, _1>;

// This probe stores the raw accumulator as BF16.  No C tensor, residual, gate,
// or auxiliary load is present, so differences between the two exposed entry
// points are GEMM scheduling/layout differences rather than epilogue work.
using FusionOperation = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90AccFetch>;

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
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
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

Element const* bf16_ptr(torch::Tensor const& t) {
  return reinterpret_cast<Element const*>(t.data_ptr<at::BFloat16>());
}

Element* mutable_bf16_ptr(torch::Tensor const& t) {
  return reinterpret_cast<Element*>(t.data_ptr<at::BFloat16>());
}

void check_inputs(torch::Tensor const& b, torch::Tensor const& weight) {
  TORCH_CHECK(b.is_cuda() && weight.is_cuda(), "all tensors must be CUDA");
  TORCH_CHECK(b.scalar_type() == at::kBFloat16, "b must be BF16");
  TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "weight must be BF16");
  TORCH_CHECK(b.dim() == 3 && weight.dim() == 2, "unexpected rank");
  TORCH_CHECK(weight.size(1) == b.size(2), "hidden mismatch");
  TORCH_CHECK(b.is_contiguous() && weight.is_contiguous(), "all tensors must be contiguous");
  TORCH_CHECK(b.size(2) % 8 == 0 && weight.size(0) % 8 == 0, "BF16 tensor-core dimensions must be aligned");
}

torch::Tensor run_gemm(
    torch::Tensor const& b,
    torch::Tensor const& weight,
    ProblemShapeType problem_shape,
    StrideA stride_a,
    StrideB stride_b,
    StrideD stride_d) {
  auto out = torch::empty({b.size(0), b.size(1), weight.size(0)}, b.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel>(
      b.get_device(), 0, 0, stream);

  MainloopArguments mainloop_args{bf16_ptr(b), stride_a, bf16_ptr(weight), stride_b};
  EpilogueArguments epilogue_args{
      {},
      nullptr,
      StrideC{},
      mutable_bf16_ptr(out),
      stride_d};
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
      torch::TensorOptions().dtype(torch::kUInt8).device(b.device()));

  cutlass::Status status = gemm.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS can_implement failed");
  status = gemm.initialize(arguments, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS initialize failed");
  status = gemm.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS run failed");
  return out;
}

}  // namespace

torch::Tensor forward_flat(torch::Tensor b, torch::Tensor weight) {
  check_inputs(b, weight);
  int64_t samples = b.size(0);
  int64_t tokens = b.size(1);
  int64_t hidden = b.size(2);
  int64_t output_channels = weight.size(0);
  ProblemShapeType problem_shape{
      static_cast<int>(samples * tokens),
      static_cast<int>(output_channels),
      static_cast<int>(hidden),
      1};
  return run_gemm(
      b,
      weight,
      problem_shape,
      StrideA{hidden, _1{}, 0},
      StrideB{hidden, _1{}, 0},
      StrideD{output_channels, _1{}, 0});
}

torch::Tensor forward_token_batched(torch::Tensor b, torch::Tensor weight) {
  check_inputs(b, weight);
  int64_t samples = b.size(0);
  int64_t tokens = b.size(1);
  int64_t hidden = b.size(2);
  int64_t output_channels = weight.size(0);
  ProblemShapeType problem_shape{
      static_cast<int>(samples),
      static_cast<int>(output_channels),
      static_cast<int>(hidden),
      static_cast<int>(tokens)};
  return run_gemm(
      b,
      weight,
      problem_shape,
      StrideA{tokens * hidden, _1{}, hidden},
      StrideB{hidden, _1{}, 0},
      StrideD{tokens * output_channels, _1{}, output_channels});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_flat", &forward_flat, "Flattened transition output GEMM CUTLASS probe");
  m.def("forward_token_batched", &forward_token_batched, "Token-batched transition output GEMM CUTLASS probe");
}
