# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings
from contextlib import contextmanager, nullcontext
from typing import Optional, Union

import torch
import torch.nn as nn

from protenix.model.modules.embedders import FourierEmbedding, RelativePositionEncoding
from protenix.model.modules.primitives import LinearNoBias, Transition
from protenix.model.modules.transformer import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    DiffusionTransformer,
)
from protenix.model.triangular.layers import LayerNorm
from protenix.model.utils import expand_at_dim, get_checkpoint_fn, permute_final_dims


_FALSE_ENV_VALUES = {"0", "false", "off", "no"}
_AUTO_ENV_VALUES = {"", "auto", "default"}


def broadcast_diffusion_s_enabled() -> bool:
    return (
        os.getenv("PROTENIX_BROADCAST_DIFFUSION_S", "0").lower()
        not in _FALSE_ENV_VALUES
    )


def _cuda_bf16_supported() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except (AssertionError, RuntimeError):
        return False


def diffusion_core_bf16_enabled() -> bool:
    """Return whether diffusion transformer activations should use BF16.

    On H100, the mixed-sequence N_sample=5 gate showed that the diffusion
    transformer was dominated by FP32/TF32 GEMMs.  Keeping its activations in
    BF16 cut that subrange by about 40% without moving pairformer or confidence
    time, so BF16-capable CUDA inference now takes the fast path by default.
    Set ``PROTENIX_BF16_DIFFUSION_CORE=0`` to recover the old conservative FP32
    core for numerical audits or unsupported deployments.
    """

    value = os.getenv("PROTENIX_BF16_DIFFUSION_CORE")
    if value is None or value.strip().lower() in _AUTO_ENV_VALUES:
        return _cuda_bf16_supported()
    return value.strip().lower() not in _FALSE_ENV_VALUES


def atom_attention_bf16_enabled() -> bool:
    """Return whether atom encoder/decoder attention should run under BF16.

    The mixed-length `N_sample=5` gate showed that BF16 atom attention by
    itself is not enough: it only becomes an end-to-end win when paired with the
    Triton local-attention path, which can consume and produce BF16 without the
    old SDPA boundary traffic.  The policy still lives here because this module
    owns the atom encoder/decoder autocast boundary.  Unsupported or non-CUDA
    devices keep the original FP32 path; set ``PROTENIX_BF16_ATOM_ATTENTION=0``
    to force the conservative path during numerical audits.
    """

    value = os.getenv("PROTENIX_BF16_ATOM_ATTENTION")
    if value is None or value.strip().lower() in _AUTO_ENV_VALUES:
        return _cuda_bf16_supported()
    return value.strip().lower() not in _FALSE_ENV_VALUES


def compile_diffusion_transformer_enabled() -> bool:
    """Return whether to compile the diffusion token transformer.

    The flattened low-sample inference path calls the same 24-block token
    transformer hundreds of times during `N_step=200` sampling.  Isolated H100
    screens show that `torch.compile` can fuse enough launch-heavy plumbing to
    move that boundary by about 1.3x, but each new padded token length pays a
    large compile cost.  Keep this opt-in until a warmed representative gate
    proves that a real campaign reuses enough shapes to amortize compilation.
    """

    value = os.getenv("PROTENIX_COMPILE_DIFFUSION_TRANSFORMER", "0")
    return value.strip().lower() not in _FALSE_ENV_VALUES


def compile_diffusion_transformer_mode() -> str:
    mode = os.getenv("PROTENIX_COMPILE_DIFFUSION_TRANSFORMER_MODE", "default")
    return mode if mode else "default"


def compile_diffusion_transformer_fullgraph() -> bool:
    value = os.getenv("PROTENIX_COMPILE_DIFFUSION_TRANSFORMER_FULLGRAPH", "0")
    return value.strip().lower() not in _FALSE_ENV_VALUES


def compile_diffusion_transformer_disable_cudnn_sdpa() -> bool:
    """Return whether compiled transformer calls should avoid cuDNN SDPA.

    The first integrated compile gate failed when Inductor selected a cuDNN SDPA
    execution plan that the runtime could not build.  Disabling only cuDNN SDPA
    preserves Flash/Efficient/Math SDPA fallbacks and kept the isolated compile
    speedup at both mixed-batch token sizes.
    """

    value = os.getenv("PROTENIX_COMPILE_DIFFUSION_DISABLE_CUDNN_SDPA", "1")
    return value.strip().lower() not in _FALSE_ENV_VALUES


def _can_broadcast_diffusion_s(
    module: nn.Module,
    t_hat_noise_level: torch.Tensor,
) -> bool:
    """Return True when diffusion conditioning is identical for every sample.

    In sampling, all structures in the same denoising step usually share the
    same scalar noise level.  If that scalar was expanded to ``N_sample`` lanes,
    the conditioning block would redo identical work for each sample.  Keeping
    a singleton sample lane lets later tensor ops broadcast the result instead.
    """
    if not broadcast_diffusion_s_enabled():
        return False
    if module.training or torch.is_grad_enabled():
        return False
    if t_hat_noise_level.size(-1) <= 1:
        return False
    if t_hat_noise_level.stride(-1) == 0:
        return True
    return bool(
        torch.all(t_hat_noise_level == t_hat_noise_level[..., :1]).item()
    )


class _CudaEventTimer:
    def __init__(
        self,
        events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]],
        name: str,
    ) -> None:
        self.events = events
        self.name = name

    def __enter__(self) -> None:
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def __exit__(self, *args: object) -> None:
        self.end.record()
        self.events.setdefault(self.name, []).append((self.start, self.end))


class DiffusionConditioning(nn.Module):
    """
    Implements Algorithm 21 in AF3

    Args:
        sigma_data (float, optional): the standard deviation of the data. Defaults to 16.0.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
        c_s_inputs (int, optional): input embedding dim from InputEmbedder. Defaults to 449.
        c_noise_embedding (int, optional): noise embedding dim. Defaults to 256.
    """

    def __init__(
        self,
        sigma_data: float = 16.0,
        c_z: int = 128,
        c_s: int = 384,
        c_s_inputs: int = 449,
        c_noise_embedding: int = 256,
    ) -> None:
        super(DiffusionConditioning, self).__init__()
        self.sigma_data = sigma_data
        self.c_z = c_z
        self.c_s = c_s
        self.c_s_inputs = c_s_inputs
        # Line1-Line3:
        self.relpe = RelativePositionEncoding(c_z=c_z)
        self.layernorm_z = LayerNorm(2 * self.c_z, create_offset=False)
        self.linear_no_bias_z = LinearNoBias(
            in_features=2 * self.c_z, out_features=self.c_z, precision=torch.float32
        )
        # Line3-Line5:
        self.transition_z1 = Transition(c_in=self.c_z, n=2)
        self.transition_z2 = Transition(c_in=self.c_z, n=2)

        # Line6-Line7
        self.layernorm_s = LayerNorm(self.c_s + self.c_s_inputs, create_offset=False)
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s + self.c_s_inputs,
            out_features=self.c_s,
            precision=torch.float32,
        )
        # Line8-Line9
        self.fourier_embedding = FourierEmbedding(c=c_noise_embedding)
        self.layernorm_n = LayerNorm(c_noise_embedding, create_offset=False)
        self.linear_no_bias_n = LinearNoBias(
            in_features=c_noise_embedding,
            out_features=self.c_s,
            precision=torch.float32,
        )
        # Line10-Line12
        self.transition_s1 = Transition(c_in=self.c_s, n=2)
        self.transition_s2 = Transition(c_in=self.c_s, n=2)
        print(f"Diffusion Module has {self.sigma_data}")

    def prepare_cache(
        self,
        relp_feature: torch.Tensor,
        z_trunk: torch.Tensor,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        # Pair conditioning
        pair_z = torch.cat(
            tensors=[
                z_trunk,
                self.relpe(relp_feature),
            ],
            dim=-1,
        )  # [..., N_tokens, N_tokens, 2*c_z]
        pair_z = self.linear_no_bias_z(self.layernorm_z(pair_z))
        if inplace_safe:
            pair_z += self.transition_z1(pair_z)
            pair_z += self.transition_z2(pair_z)
        else:
            pair_z = pair_z + self.transition_z1(pair_z)
            pair_z = pair_z + self.transition_z2(pair_z)
        return pair_z

    def forward(
        self,
        t_hat_noise_level: torch.Tensor,
        relp_feature: torch.Tensor,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        pair_z: torch.Tensor,
        inplace_safe: bool = False,
        use_conditioning: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            t_hat_noise_level (torch.Tensor): the noise level
                [..., N_sample]
            asym_id (torch.Tensor): asym_id
            residue_index (torch.Tensor): residue_index
            entity_id (torch.Tensor): entity_id
            token_index (torch.Tensor): token_index
            sym_id (torch.Tensor): sym_id
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            inplace_safe (bool): Whether it is safe to use inplace operations.
            use_conditioning (bool): Whether to drop the s/z embeddings.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: embeddings s and z
                - s (torch.Tensor): [..., N_sample, N_tokens, c_s]
                - z (torch.Tensor): [..., N_tokens, N_tokens, c_z]
        """
        if pair_z is None:
            if not use_conditioning:
                if inplace_safe:
                    s_trunk *= 0
                    z_trunk *= 0
                else:
                    s_trunk = 0 * s_trunk
                    z_trunk = 0 * z_trunk
            pair_z = self.prepare_cache(relp_feature, z_trunk, inplace_safe)
        else:
            # Pair conditioning
            if inplace_safe:
                pair_z_clone = pair_z.clone()
                pair_z = pair_z_clone
        # Single conditioning
        single_s = torch.cat(
            tensors=[s_trunk, s_inputs], dim=-1
        )  # [..., N_tokens, c_s + c_s_inputs]
        single_s = self.linear_no_bias_s(self.layernorm_s(single_s))
        noise_n = self.fourier_embedding(
            t_hat_noise_level=torch.log(input=t_hat_noise_level / self.sigma_data) / 4
        ).to(
            single_s.dtype
        )  # [..., N_sample, c_in]
        single_s = single_s.unsqueeze(dim=-3) + self.linear_no_bias_n(
            self.layernorm_n(noise_n)
        ).unsqueeze(
            dim=-2
        )  # [..., N_sample, N_tokens, c_s]
        if inplace_safe:
            single_s += self.transition_s1(single_s)
            single_s += self.transition_s2(single_s)
        else:
            single_s = single_s + self.transition_s1(single_s)
            single_s = single_s + self.transition_s2(single_s)
        return single_s, pair_z


class DiffusionSchedule:
    """
    Diffusion schedule for training and inference.

    Args:
        sigma_data (float, optional): The standard deviation of the data. Defaults to 16.0.
        s_max (float, optional): The maximum noise level. Defaults to 160.0.
        s_min (float, optional): The minimum noise level. Defaults to 4e-4.
        p (float, optional): The exponent for the noise schedule. Defaults to 7.0.
        dt (float, optional): The time step size. Defaults to 1/200.
        p_mean (float, optional): The mean of the log-normal distribution for noise level sampling. Defaults to -1.2.
        p_std (float, optional): The standard deviation of the log-normal distribution for noise level sampling. Defaults to 1.5.
    """

    def __init__(
        self,
        sigma_data: float = 16.0,
        s_max: float = 160.0,
        s_min: float = 4e-4,
        p: float = 7.0,
        dt: float = 1 / 200,
        p_mean: float = -1.2,
        p_std: float = 1.5,
    ) -> None:
        self.sigma_data = sigma_data
        self.s_max = s_max
        self.s_min = s_min
        self.p = p
        self.dt = dt
        self.p_mean = p_mean
        self.p_std = p_std
        # self.T
        self.T = int(1 / dt) + 1  # 201

    def get_train_noise_schedule(self) -> torch.Tensor:
        return self.sigma_data * torch.exp(self.p_mean + self.p_std * torch.randn(1))

    def get_inference_noise_schedule(self) -> torch.Tensor:
        time_step_lists = torch.arange(start=0, end=1 + 1e-10, step=self.dt)
        inference_noise_schedule = (
            self.sigma_data
            * (
                self.s_max ** (1 / self.p)
                + time_step_lists
                * (self.s_min ** (1 / self.p) - self.s_max ** (1 / self.p))
            )
            ** self.p
        )
        return inference_noise_schedule


class DiffusionModule(nn.Module):
    """
    Implements Algorithm 20 in AF3

    Args:
        sigma_data (torch.float, optional): the standard deviation of the data. Defaults to 16.0.
        c_atom (int, optional): embedding dim for atom feature. Defaults to 128.
        c_atompair (int, optional): embedding dim for atompair feature. Defaults to 16.
        c_token (int, optional): feature channel of token (single a). Defaults to 768.
        c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        c_s_inputs (int, optional): hidden dim [for single input embedding]. Defaults to 449.
        atom_encoder (dict[str, int], optional): configs in AtomAttentionEncoder. Defaults to {"n_blocks": 3, "n_heads": 4}.
        transformer (dict[str, int], optional): configs in DiffusionTransformer. Defaults to {"n_blocks": 24, "n_heads": 16}.
        atom_decoder (dict[str, int], optional): configs in AtomAttentionDecoder. Defaults to {"n_blocks": 3, "n_heads": 4}.
        drop_path_rate (float, optional): drop path rate. Defaults to 0.0.
        blocks_per_ckpt: number of atom_encoder/transformer/atom_decoder blocks in each activation checkpoint
            Size of each chunk. A higher value corresponds to fewer
            checkpoints, and trades memory for speed. If None, no checkpointing is performed.
        use_fine_grained_checkpoint: whether use fine-gained checkpoint for finetuning stage 2
            only effective if blocks_per_ckpt is not None.
    """

    def __init__(
        self,
        sigma_data: float = 16.0,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_token: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        c_s_inputs: int = 449,
        atom_encoder: dict[str, int] = {"n_blocks": 3, "n_heads": 4},
        transformer: dict[str, int] = {
            "n_blocks": 24,
            "n_heads": 16,
            "drop_path_rate": 0,
        },
        atom_decoder: dict[str, int] = {"n_blocks": 3, "n_heads": 4},
        drop_path_rate: float = 0.0,
        blocks_per_ckpt: Optional[int] = None,
        use_fine_grained_checkpoint: bool = False,
    ) -> None:
        super(DiffusionModule, self).__init__()
        self.sigma_data = sigma_data
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_s_inputs = c_s_inputs
        self.c_s = c_s
        self.c_z = c_z

        # Grad checkpoint setting
        self.blocks_per_ckpt = blocks_per_ckpt
        self.use_fine_grained_checkpoint = use_fine_grained_checkpoint

        self.diffusion_conditioning = DiffusionConditioning(
            sigma_data=self.sigma_data, c_z=c_z, c_s=c_s, c_s_inputs=c_s_inputs
        )
        self.atom_attention_encoder = AtomAttentionEncoder(
            **atom_encoder,
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            has_coords=True,
            c_s=c_s,
            c_z=c_z,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        # Alg20: line4
        self.layernorm_s = LayerNorm(c_s, create_offset=False)
        self.linear_no_bias_s = LinearNoBias(
            in_features=c_s,
            out_features=c_token,
            precision=torch.float32,
            initializer="zeros",
        )
        self.diffusion_transformer = DiffusionTransformer(
            **transformer,
            c_a=c_token,
            c_s=c_s,
            c_z=c_z,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.layernorm_a = LayerNorm(c_token, create_offset=False)
        self.atom_attention_decoder = AtomAttentionDecoder(
            **atom_decoder,
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.normalize = LayerNorm(c_z, create_offset=False, create_scale=False)
        self._perf_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        self._diffusion_transformer_compile_attempted = False
        self._diffusion_transformer_compiled = False

    def reset_perf_stats(self) -> None:
        self._perf_events = {}

    def _profile_block(self, name: str):
        if (
            os.getenv("PROTENIX_TIMING_DIFFUSION", "0").lower()
            in {"0", "false", "off", "no"}
            or not torch.cuda.is_available()
        ):
            return nullcontext()
        return _CudaEventTimer(self._perf_events, name)

    def consume_perf_stats(self) -> dict[str, float]:
        stats = {}
        for name, events in self._perf_events.items():
            total_ms = 0.0
            for start, end in events:
                end.synchronize()
                total_ms += start.elapsed_time(end)
            stats[f"diffusion_{name}_sec"] = total_ms / 1000.0
        self.reset_perf_stats()
        return stats

    def _diffusion_core_dtype(self, fallback: torch.dtype) -> torch.dtype:
        if not diffusion_core_bf16_enabled():
            return torch.float32
        if fallback == torch.float16:
            return torch.float16
        return torch.bfloat16

    def _diffusion_core_autocast(self, dtype: torch.dtype):
        if not diffusion_core_bf16_enabled() or not torch.cuda.is_available():
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=dtype)

    def maybe_compile_diffusion_transformer(self) -> None:
        """Lazily wrap the diffusion transformer with `torch.compile`.

        Wrapping in ``__init__`` would change checkpoint-loading and state-dict
        behavior.  The first inference call happens after weights have been
        loaded and the module is on its final device, which is the right point
        to install an opt-in compiled wrapper.  Actual graph specialization
        still happens on first use of each padded token shape.
        """

        if self._diffusion_transformer_compile_attempted:
            return
        if (
            not compile_diffusion_transformer_enabled()
            or self.training
            or torch.is_grad_enabled()
            or not hasattr(torch, "compile")
        ):
            return
        self._diffusion_transformer_compile_attempted = True
        try:
            self.diffusion_transformer = torch.compile(
                self.diffusion_transformer,
                mode=compile_diffusion_transformer_mode(),
                fullgraph=compile_diffusion_transformer_fullgraph(),
                dynamic=False,
            )
            self._diffusion_transformer_compiled = True
        except Exception as exc:  # pragma: no cover - runtime dependent.
            warnings.warn(
                "Could not wrap diffusion transformer with torch.compile; "
                f"continuing eagerly. Original error: {exc!r}",
                RuntimeWarning,
            )

    @contextmanager
    def _compiled_transformer_sdpa_context(self):
        if (
            not self._diffusion_transformer_compiled
            or not compile_diffusion_transformer_disable_cudnn_sdpa()
            or not torch.cuda.is_available()
            or not hasattr(torch.backends.cuda, "enable_cudnn_sdp")
            or not hasattr(torch.backends.cuda, "cudnn_sdp_enabled")
        ):
            yield
            return

        # Scope the backend change to this transformer call.  Other model
        # attention paths can keep their default backend selection, while the
        # compiled diffusion graph avoids the cuDNN SDPA plan that failed in the
        # first integrated compile gate.
        previous = torch.backends.cuda.cudnn_sdp_enabled()
        torch.backends.cuda.enable_cudnn_sdp(False)
        try:
            yield
        finally:
            torch.backends.cuda.enable_cudnn_sdp(previous)

    def _atom_attention_autocast(self):
        if not atom_attention_bf16_enabled() or not torch.cuda.is_available():
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def f_forward(
        self,
        r_noisy: torch.Tensor,
        t_hat_noise_level: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        pair_z: torch.Tensor,
        p_lm: torch.Tensor,
        c_l: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        use_conditioning: bool = True,
        enable_efficient_fusion: bool = False,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The raw network to be trained.
        As in EDM equation (7), this is F_theta(c_in * x, c_noise(sigma)).
        Here, c_noise(sigma) is computed in Conditioning module.

        Args:
            r_noisy (torch.Tensor): scaled x_noisy (i.e., c_in * x)
                [..., N_sample, N_atom, 3]
            t_hat_noise_level (torch.Tensor): the noise level, as well as the time step t
                [..., N_sample]
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input feature
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            pair_z (torch.Tensor): diffusion pair embedding
                [..., N_tokens, N_tokens, c_z]
            p_lm (torch.Tensor): MSA embedding
                [..., N_tokens, c_p_lm]
            c_l (torch.Tensor): ligand embedding
                [..., N_tokens, c_c_l]
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.
            use_conditioning (bool): Whether to drop the s/z embeddings in DiffusionConditioning.
            enable_efficient_fusion (bool): Whether to enable efficient fusion. Defaults to False.

        Returns:
            torch.Tensor: coordinates update
                [..., N_sample, N_atom, 3]
        """
        N_sample = r_noisy.size(-3)
        assert t_hat_noise_level.size(-1) == N_sample

        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None
        conditioning_noise_level = t_hat_noise_level
        if _can_broadcast_diffusion_s(self, t_hat_noise_level):
            # In inference sampling, t_hat is a scalar expanded across samples.
            # The resulting single conditioning is therefore sample-invariant;
            # keep only one sample lane and rely on broadcasting downstream.
            conditioning_noise_level = t_hat_noise_level[..., :1]
        # Conditioning, shared across difference samples
        # Diffusion_conditioning consumes 7-8G when token num is 768,
        # use checkpoint here if blocks_per_ckpt is not None.
        with self._profile_block("conditioning"), torch.profiler.record_function(
            "protenix/diffusion_conditioning"
        ):
            if blocks_per_ckpt:
                checkpoint_fn = get_checkpoint_fn()
                s_single, z_pair = checkpoint_fn(
                    self.diffusion_conditioning,
                    conditioning_noise_level,
                    input_feature_dict["relp"],
                    s_inputs,
                    s_trunk,
                    z_trunk,
                    pair_z,
                    inplace_safe,
                    use_conditioning,
                )
            else:
                s_single, z_pair = self.diffusion_conditioning(
                    conditioning_noise_level,
                    input_feature_dict["relp"],
                    s_inputs=s_inputs,
                    s_trunk=s_trunk,
                    z_trunk=z_trunk,
                    pair_z=pair_z,
                    inplace_safe=inplace_safe,
                    use_conditioning=use_conditioning,
                )  # [..., N_sample, N_token, c_s], [..., N_token, N_token, c_z]

        # Expand embeddings to match N_sample
        s_trunk = expand_at_dim(s_trunk, dim=-3, n=1)  # [..., N_sample, N_token, c_s]
        z_pair = expand_at_dim(
            z_pair, dim=-4, n=1
        )  # [..., N_sample, N_token, N_token, c_z]
        # Fine-grained checkpoint for finetuning stage 2 (token num: 768) for avoiding OOM
        with self._profile_block("atom_encoder"), torch.profiler.record_function(
            "protenix/atom_attention_encoder"
        ):
            with self._atom_attention_autocast():
                if blocks_per_ckpt and self.use_fine_grained_checkpoint:
                    checkpoint_fn = get_checkpoint_fn()
                    a_token, q_skip, c_skip, p_skip = checkpoint_fn(
                        self.atom_attention_encoder,
                        input_feature_dict["atom_to_token_idx"],
                        input_feature_dict["ref_pos"],
                        input_feature_dict["ref_charge"],
                        input_feature_dict["ref_mask"],
                        input_feature_dict["ref_atom_name_chars"],
                        input_feature_dict["ref_element"],
                        input_feature_dict["d_lm"],
                        input_feature_dict["v_lm"],
                        input_feature_dict["pad_info"],
                        r_noisy,
                        s_trunk,
                        z_pair,
                        p_lm,
                        c_l,
                        inplace_safe,
                        chunk_size,
                    )
                else:
                    # Sequence-local Atom Attention and aggregation to coarse-grained tokens
                    a_token, q_skip, c_skip, p_skip = self.atom_attention_encoder(
                        input_feature_dict["atom_to_token_idx"],
                        input_feature_dict["ref_pos"],
                        input_feature_dict["ref_charge"],
                        input_feature_dict["ref_mask"],
                        input_feature_dict["ref_atom_name_chars"],
                        input_feature_dict["ref_element"],
                        input_feature_dict["d_lm"],
                        input_feature_dict["v_lm"],
                        input_feature_dict["pad_info"],
                        r_l=r_noisy,
                        s=s_trunk,
                        z=z_pair,
                        p_lm=p_lm,
                        c_l=c_l,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
        transformer_dtype = self._diffusion_core_dtype(a_token.dtype)
        a_token = a_token.to(dtype=transformer_dtype)

        # Full self-attention on token level.
        if inplace_safe:
            a_token += self.linear_no_bias_s(
                self.layernorm_s(s_single)
            )  # [..., N_sample, N_token, c_token]
        else:
            a_token = a_token + self.linear_no_bias_s(
                self.layernorm_s(s_single)
            )  # [..., N_sample, N_token, c_token]
        if enable_efficient_fusion:
            z = self.normalize(z_pair.to(dtype=transformer_dtype))
            z = permute_final_dims(z, [2, 0, 1]).contiguous()
        else:
            z = z_pair.to(dtype=transformer_dtype)
        with self._profile_block("transformer"), torch.profiler.record_function(
            "protenix/diffusion_transformer"
        ):
            self.maybe_compile_diffusion_transformer()
            with (
                self._compiled_transformer_sdpa_context(),
                self._diffusion_core_autocast(transformer_dtype),
            ):
                a_token = self.diffusion_transformer(
                    a=a_token,
                    s=s_single.to(dtype=transformer_dtype),
                    z=z,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                    enable_efficient_fusion=enable_efficient_fusion,
                    token_mask=token_mask,
                )
        if a_token.dtype != torch.float32:
            a_token = a_token.to(dtype=torch.float32)

        a_token = self.layernorm_a(a_token)

        # Fine-grained checkpoint for finetuning stage 2 (token num: 768) for avoiding OOM
        with self._profile_block("atom_decoder"), torch.profiler.record_function(
            "protenix/atom_attention_decoder"
        ):
            with self._atom_attention_autocast():
                if blocks_per_ckpt and self.use_fine_grained_checkpoint:
                    checkpoint_fn = get_checkpoint_fn()
                    r_update = checkpoint_fn(
                        self.atom_attention_decoder,
                        input_feature_dict["atom_to_token_idx"],
                        a_token,
                        q_skip,
                        c_skip,
                        p_skip,
                        inplace_safe,
                        chunk_size,
                    )
                else:
                    # Broadcast token activations to atoms and run Sequence-local Atom Attention
                    r_update = self.atom_attention_decoder(
                        atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
                        a=a_token,
                        q_skip=q_skip,
                        c_skip=c_skip,
                        p_skip=p_skip,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )

        return r_update

    def forward(
        self,
        x_noisy: torch.Tensor,
        t_hat_noise_level: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        pair_z: torch.Tensor,
        p_lm: torch.Tensor,
        c_l: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        use_conditioning: bool = True,
        enable_efficient_fusion: bool = False,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One step denoise: x_noisy, noise_level -> x_denoised

        Args:
            x_noisy (torch.Tensor): the noisy version of the input atom coords
                [..., N_sample, N_atom,3]
            t_hat_noise_level (torch.Tensor): the noise level, as well as the time step t
                [..., N_sample]
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input meta feature dict
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            pair_z (torch.Tensor): diffusion pair embedding
                [..., N_tokens, N_tokens, c_z]
            p_lm (torch.Tensor): MSA embedding
                [..., N_tokens, c_p_lm]
            c_l (torch.Tensor): ligand embedding
                [..., N_tokens, c_c_l]
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.
            use_conditioning (bool): Whether to drop the s/z embeddings in DiffusionConditioning.
            enable_efficient_fusion (bool): Whether to enable efficient fusion. Defaults to False.

        Returns:
            torch.Tensor: the denoised coordinates of x
                [..., N_sample, N_atom,3]
        """
        # Scale positions to dimensionless vectors with approximately unit variance
        # As in EDM:
        #     r_noisy = (c_in * x_noisy)
        #     where c_in = 1 / sqrt(sigma_data^2 + sigma^2)
        with self._profile_block("input_scale"):
            r_noisy = (
                x_noisy
                / torch.sqrt(self.sigma_data**2 + t_hat_noise_level**2)[..., None, None]
            )

        # Compute the update given r_noisy (the scaled x_noisy)
        # As in EDM:
        #     r_update = F(r_noisy, c_noise(sigma))
        r_update = self.f_forward(
            r_noisy=r_noisy,
            t_hat_noise_level=t_hat_noise_level,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_z=pair_z,
            p_lm=p_lm,
            c_l=c_l,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            use_conditioning=use_conditioning,
            enable_efficient_fusion=enable_efficient_fusion,
            token_mask=token_mask,
        )

        # Rescale updates to positions and combine with input positions
        # As in EDM:
        #     D = c_skip * x_noisy + c_out * r_update
        #     c_skip = sigma_data^2 / (sigma_data^2 + sigma^2)
        #     c_out = (sigma_data * sigma) / sqrt(sigma_data^2 + sigma^2)
        #     s_ratio = sigma / sigma_data
        #     c_skip = 1 / (1 + s_ratio^2)
        #     c_out = sigma / sqrt(1 + s_ratio^2)

        with self._profile_block("output_rescale"):
            s_ratio = (t_hat_noise_level / self.sigma_data)[..., None, None].to(
                r_update.dtype
            )
            x_denoised = (
                1 / (1 + s_ratio**2) * x_noisy
                + t_hat_noise_level[..., None, None]
                / torch.sqrt(1 + s_ratio**2)
                * r_update
            ).to(r_update.dtype)

        return x_denoised
