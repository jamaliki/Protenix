#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cute/tensor.hpp>

#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp>
#include <cutlass/epilogue/thread/activation.h>
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
using StrideGate = Stride<_0, _1, int64_t>;

using ResidualScale = cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementCompute>;
using GateLoad = cutlass::epilogue::fusion::Sm90AuxLoad<
    0,
    cutlass::epilogue::collective::EpilogueTileAuto,
    Element,
    StrideGate,
    void,
    void,
    8>;

// D = residual + sigmoid(gate) * acc
//
// CUTLASS epilogue visitor trees are easiest to reason about from the leaves up:
//   1. GateLoad reads gate[n, l] with stride_m=0 so it broadcasts over samples.
//   2. GateSigmoid applies sigmoid(gate).
//   3. GateTimesAcc multiplies that value by the GEMM accumulator.
//   4. GateResidualEVT adds the residual C tensor with scale 1.0.
//
// The final tree computes exactly the PyTorch hotspot:
//   residual + sigmoid(gate) * (b @ weight.T)
using GateSigmoid = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<cutlass::epilogue::thread::Sigmoid, ElementCompute, ElementCompute, RoundStyle>,
    GateLoad>;

using GateTimesAcc = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>,
    GateSigmoid,
    cutlass::epilogue::fusion::Sm90AccFetch>;

using GateResidualEVT = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<cutlass::homogeneous_multiply_add, Element, ElementCompute, RoundStyle>,
    ResidualScale,
    cutlass::epilogue::fusion::Sm90SrcFetch<Element>,
    GateTimesAcc>;

using ResidualScaleArguments = typename ResidualScale::Arguments;
using GateLoadArguments = typename GateLoad::Arguments;
using GateSigmoidArguments = typename GateSigmoid::Arguments;
using GateTimesAccArguments = typename GateTimesAcc::Arguments;
using GateResidualArguments = typename GateResidualEVT::Arguments;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    TileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementCompute,
    Element,
    cutlass::layout::RowMajor,
    8,
    Element,
    cutlass::layout::RowMajor,
    8,
    cutlass::epilogue::TmaWarpSpecialized,
    GateResidualEVT>::CollectiveOp;

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

Element const* gate_bf16_ptr(torch::Tensor const& t) {
  return reinterpret_cast<Element const*>(t.data_ptr<at::BFloat16>());
}

GateResidualArguments make_gate_residual_args(torch::Tensor const& gate, int64_t output_channels) {
  // The tree argument layout mirrors the type definition above:
  // {residual_scale, residual_fetch, {sigmoid_gate, accumulator_fetch, multiply}, add}.
  // Using typed sub-arguments avoids the unreadable nested-brace errors that
  // otherwise come from CUTLASS's aggregate visitor-tree API.
  return GateResidualArguments{
      ResidualScaleArguments{{ElementCompute(1.0f)}},
      {},
      GateTimesAccArguments{
          GateSigmoidArguments{
              GateLoadArguments{gate_bf16_ptr(gate), Element(0), StrideGate{_0{}, _1{}, output_channels}},
              {}},
          {},
          {}},
      {}};
}

void check_inputs(
    torch::Tensor const& b,
    torch::Tensor const& weight,
    torch::Tensor const& gate,
    torch::Tensor const& residual) {
  TORCH_CHECK(b.is_cuda() && weight.is_cuda() && gate.is_cuda() && residual.is_cuda(), "all tensors must be CUDA");
  TORCH_CHECK(b.scalar_type() == at::kBFloat16, "b must be BF16");
  TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "weight must be BF16");
  TORCH_CHECK(gate.scalar_type() == at::kBFloat16, "gate must be BF16");
  TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be BF16");
  TORCH_CHECK(b.dim() == 3 && weight.dim() == 2 && gate.dim() == 3 && residual.dim() == 3, "unexpected rank");
  TORCH_CHECK(gate.size(0) == 1, "only gate_batch=1 is supported");
  TORCH_CHECK(b.size(0) == residual.size(0), "sample mismatch");
  TORCH_CHECK(b.size(1) == residual.size(1) && b.size(1) == gate.size(1), "token mismatch");
  TORCH_CHECK(weight.size(0) == residual.size(2) && weight.size(0) == gate.size(2), "output-channel mismatch");
  TORCH_CHECK(weight.size(1) == b.size(2), "hidden mismatch");
  TORCH_CHECK(b.is_contiguous() && weight.is_contiguous() && gate.is_contiguous() && residual.is_contiguous(), "all tensors must be contiguous");
  TORCH_CHECK(b.size(2) % 8 == 0 && weight.size(0) % 8 == 0, "BF16 tensor-core dimensions must be aligned");
}

}  // namespace

torch::Tensor forward(
    torch::Tensor b,
    torch::Tensor weight,
    torch::Tensor gate,
    torch::Tensor residual) {
  check_inputs(b, weight, gate, residual);

  int64_t samples = b.size(0);
  int64_t tokens = b.size(1);
  int64_t hidden = b.size(2);
  int64_t output_channels = weight.size(0);
  auto out = torch::empty_like(residual);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel>(
      b.get_device(), 0, 0, stream);
  ProblemShapeType problem_shape{
      static_cast<int>(samples),
      static_cast<int>(output_channels),
      static_cast<int>(hidden),
      static_cast<int>(tokens)};
  MainloopArguments mainloop_args{
      bf16_ptr(b), StrideA{tokens * hidden, _1{}, hidden},
      bf16_ptr(weight), StrideB{hidden, _1{}, 0}};
  EpilogueArguments epilogue_args{
      make_gate_residual_args(gate, output_channels),
      bf16_ptr(residual), StrideC{tokens * output_channels, _1{}, output_channels},
      mutable_bf16_ptr(out), StrideD{tokens * output_channels, _1{}, output_channels}};

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      mainloop_args,
      epilogue_args,
      hw_info,
      {}};

  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = torch::empty({static_cast<long>(workspace_size)}, torch::TensorOptions().dtype(torch::kUInt8).device(b.device()));

  cutlass::Status status = gemm.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS can_implement failed");
  status = gemm.initialize(arguments, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS initialize failed");
  status = gemm.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS run failed");
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Transition output epilogue CUTLASS SM90 candidate");
}
