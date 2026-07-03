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

import copy
import os
import random
import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from protenix.model import sample_confidence
from protenix.model.generator import (
    InferenceNoiseScheduler,
    sample_diffusion,
    sample_diffusion_training,
    TrainingNoiseSampler,
)
from protenix.model.modules.confidence import ConfidenceHead
from protenix.model.modules.diffusion import DiffusionModule
from protenix.model.modules.embedders import (
    ConstraintEmbedder,
    InputFeatureEmbedder,
    RelativePositionEncoding,
)
from protenix.model.modules.head import DistogramHead
from protenix.model.modules.pairformer import (
    MSAModule,
    PairformerStack,
    TemplateEmbedder,
)
from protenix.model.modules.primitives import LinearNoBias
from protenix.model.triangular.layers import LayerNorm
from protenix.model.utils import (
    centre_random_augmentation,
    permute_final_dims,
    simple_merge_dict_list,
)
from protenix.utils.logger import get_logger
from protenix.utils.permutation.permutation import SymmetricPermutation
from protenix.utils.torch_utils import autocasting_disable_decorator

logger = get_logger(__name__)


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_flag_enabled_by_default(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return int(value)


def _sync_cuda_for_timing(enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()


def _stack_padded_token_vectors(
    tensors: list[torch.Tensor],
    max_tokens: int,
) -> torch.Tensor:
    """Stack ``[N_token, C]`` tensors while keeping padding explicit.

    Padding is safe only because the downstream token transformer receives the
    matching key mask.  Fake token activations are zero-filled, but valid tokens
    keep exactly the values produced by the ragged atom encoder.
    """
    batch = len(tensors)
    first = tensors[0]
    padded = first.new_zeros((batch, max_tokens, *first.shape[1:]))
    for batch_idx, tensor in enumerate(tensors):
        padded[batch_idx, : tensor.shape[0]] = tensor
    return padded


def _stack_padded_token_pairs(
    tensors: list[torch.Tensor],
    max_tokens: int,
    channel_first: bool,
) -> torch.Tensor:
    """Stack token-pair tensors in either ``[N, N, C]`` or ``[C, N, N]`` form."""
    batch = len(tensors)
    first = tensors[0]
    if channel_first:
        padded = first.new_zeros((batch, first.shape[0], max_tokens, max_tokens))
        for batch_idx, tensor in enumerate(tensors):
            n_token = tensor.shape[-1]
            padded[batch_idx, :, :n_token, :n_token] = tensor
    else:
        padded = first.new_zeros((batch, max_tokens, max_tokens, first.shape[-1]))
        for batch_idx, tensor in enumerate(tensors):
            n_token = tensor.shape[0]
            padded[batch_idx, :n_token, :n_token] = tensor
    return padded


def _stack_padded_axis(
    tensors: list[torch.Tensor],
    axis: int,
    max_size: int,
) -> torch.Tensor:
    """Stack tensors after zero-padding one ragged axis to ``max_size``."""
    first = tensors[0]
    axis = axis if axis >= 0 else first.dim() + axis
    out_shape = [len(tensors), *first.shape]
    out_shape[1 + axis] = max_size
    padded = first.new_zeros(out_shape)
    for batch_idx, tensor in enumerate(tensors):
        slices = [batch_idx]
        for size in tensor.shape:
            slices.append(slice(0, size))
        padded[tuple(slices)] = tensor
    return padded


def _stack_padded_atom_indices(
    tensors: list[torch.Tensor],
    max_atoms: int,
) -> torch.Tensor:
    padded = tensors[0].new_zeros((len(tensors), max_atoms))
    for batch_idx, tensor in enumerate(tensors):
        padded[batch_idx, : tensor.shape[0]] = tensor
    return padded


def _make_token_mask(
    token_counts: list[int],
    max_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    token_mask = torch.zeros(
        (len(token_counts), max_tokens), device=device, dtype=torch.bool
    )
    for batch_idx, n_token in enumerate(token_counts):
        token_mask[batch_idx, :n_token] = True
    return token_mask


def _make_atom_mask(
    atom_counts: list[int],
    max_atoms: int,
    device: torch.device,
) -> torch.Tensor:
    atom_mask = torch.zeros(
        (len(atom_counts), max_atoms), device=device, dtype=torch.bool
    )
    for batch_idx, n_atom in enumerate(atom_counts):
        atom_mask[batch_idx, :n_atom] = True
    return atom_mask


def _expand_sample_axis(tensor: torch.Tensor, n_sample: int) -> torch.Tensor:
    """Expand a ``[B, 1, ...]`` diffusion tensor over low sample counts.

    In the inference sampler every record uses the same scalar noise level at a
    denoising step.  Conditioning tensors are therefore sample-invariant.  Keep
    them as one lane while building them, then broadcast here before the token
    transformer needs one row per ``(record, sample)``.  This is a view when the
    sample axis is singleton, so it avoids recomputing and storing identical
    conditioning activations.
    """
    if tensor.shape[1] == n_sample:
        return tensor
    assert tensor.shape[1] == 1
    return tensor.expand(tensor.shape[0], n_sample, *tensor.shape[2:])


def _flatten_record_sample_axes(tensor: torch.Tensor, n_sample: int) -> torch.Tensor:
    """Flatten ``[B, N_sample, ...]`` to ``[B * N_sample, ...]``.

    The token diffusion transformer can operate on arbitrary leading batch
    items, but the attention helper deliberately switches to a conservative
    FP32 path when q/k/v have two leading axes.  Flattening record and sample
    keeps the fast rank-4 BF16/cuDNN path while still sharing one launch across
    all proteins and samples.
    """
    tensor = _expand_sample_axis(tensor, n_sample)
    return tensor.reshape(tensor.shape[0] * n_sample, *tensor.shape[2:])


def _flatten_record_sample_mask(
    mask: torch.Tensor,
    n_sample: int,
) -> torch.Tensor:
    return mask[:, None, :].expand(mask.shape[0], n_sample, mask.shape[1]).reshape(
        mask.shape[0] * n_sample, mask.shape[1]
    )


def _diffusion_transformer_sample_axis_enabled() -> bool:
    return _env_flag_enabled("PROTENIX_DIFFUSION_TRANSFORMER_SAMPLE_AXIS")


def update_input_feature_dict(input_feature_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Lines 1-3 of Algorithm 5 compute d_lm, v_lm, and pad_info utilized in the AtomAttentionEncoder.
    Args:
            input_feature_dict (dict[str, Any]): input features
    Returns:
            input_feature_dict (dict[str, Any]): input features
    """
    from protenix.model.modules.transformer import rearrange_qk_to_dense_trunk

    with torch.no_grad():
        # Prepare tensors in dense trunks for local operations
        q_trunked_list, k_trunked_list, pad_info = rearrange_qk_to_dense_trunk(
            q=[input_feature_dict["ref_pos"], input_feature_dict["ref_space_uid"]],
            k=[input_feature_dict["ref_pos"], input_feature_dict["ref_space_uid"]],
            dim_q=[-2, -1],
            dim_k=[-2, -1],
            n_queries=32,
            n_keys=128,
            compute_mask=True,
        )
        # Compute atom pair feature
        d_lm = (
            q_trunked_list[0][..., None, :] - k_trunked_list[0][..., None, :, :]
        )  # [..., n_blocks, n_queries, n_keys, 3]
        v_lm = (
            q_trunked_list[1][..., None].int() == k_trunked_list[1][..., None, :].int()
        ).unsqueeze(
            dim=-1
        )  # [..., n_blocks, n_queries, n_keys, 1]
        input_feature_dict["d_lm"] = d_lm
        input_feature_dict["v_lm"] = v_lm
        input_feature_dict["pad_info"] = pad_info
        return input_feature_dict


class Protenix(nn.Module):
    """
    Implements Algorithm 1 [Main Inference/Train Loop] in AF3
    """

    def __init__(self, configs: Any) -> None:
        super(Protenix, self).__init__()
        self.configs = configs
        torch.backends.cuda.matmul.allow_tf32 = self.configs.enable_tf32
        # Some constants
        self.enable_diffusion_shared_vars_cache = (
            self.configs.enable_diffusion_shared_vars_cache
        )
        self.enable_efficient_fusion = self.configs.enable_efficient_fusion
        self.N_cycle = self.configs.model.N_cycle
        self.N_model_seed = self.configs.model.N_model_seed
        self.train_confidence_only = configs.train_confidence_only
        if self.train_confidence_only:  # the final finetune stage
            assert configs.loss.weight.alpha_diffusion == 0.0
            assert configs.loss.weight.alpha_distogram == 0.0

        # Diffusion scheduler
        self.train_noise_sampler = TrainingNoiseSampler(**configs.train_noise_sampler)
        self.inference_noise_scheduler = InferenceNoiseScheduler(
            **configs.inference_noise_scheduler
        )
        self.diffusion_batch_size = self.configs.diffusion_batch_size

        # Model
        esm_configs = configs.get("esm", {})  # This is used in InputFeatureEmbedder
        self.input_embedder = InputFeatureEmbedder(
            **configs.model.input_embedder, esm_configs=esm_configs
        )
        self.relative_position_encoding = RelativePositionEncoding(
            **configs.model.relative_position_encoding
        )
        self.template_embedder = TemplateEmbedder(**configs.model.template_embedder)
        self.msa_module = MSAModule(
            **configs.model.msa_module,
            msa_configs=configs.data.get("msa", {}),
        )
        self.constraint_embedder = ConstraintEmbedder(
            **configs.model.constraint_embedder
        )
        self.pairformer_stack = PairformerStack(**configs.model.pairformer)
        self.diffusion_module = DiffusionModule(**configs.model.diffusion_module)
        self.distogram_head = DistogramHead(**configs.model.distogram_head)
        self.confidence_head = ConfidenceHead(**configs.model.confidence_head)

        self.c_s, self.c_z, self.c_s_inputs = (
            configs.c_s,
            configs.c_z,
            configs.c_s_inputs,
        )
        self.linear_no_bias_sinit = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_s
        )
        self.linear_no_bias_zinit1 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_zinit2 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_token_bond = LinearNoBias(
            in_features=1, out_features=self.c_z
        )
        self.linear_no_bias_z_cycle = LinearNoBias(
            in_features=self.c_z, out_features=self.c_z
        )
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s, out_features=self.c_s
        )
        self.layernorm_z_cycle = LayerNorm(self.c_z)
        self.layernorm_s = LayerNorm(self.c_s)

        # Zero init the recycling layer
        nn.init.zeros_(self.linear_no_bias_z_cycle.weight)
        nn.init.zeros_(self.linear_no_bias_s.weight)

    def get_pairformer_output(
        self,
        input_feature_dict: dict[str, Any],
        N_cycle: int,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        mc_dropout: bool = False,
        mc_dropout_rate: float = 0.4,
    ) -> tuple[torch.Tensor, ...]:
        """
        The forward pass from the input to pairformer output

        Args:
            input_feature_dict (dict[str, Any]): input features
            N_cycle (int): number of cycles
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            Tuple[torch.Tensor, ...]: s_inputs, s, z
        """
        if self.train_confidence_only:
            self.input_embedder.eval()
            self.template_embedder.eval()
            self.msa_module.eval()
            self.pairformer_stack.eval()

        # Line 1-5
        s_inputs = self.input_embedder(
            input_feature_dict, inplace_safe=False, chunk_size=chunk_size
        )  # [..., N_token, 449]

        s, z = self.get_pairformer_output_from_s_inputs(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            N_cycle=N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            mc_dropout=mc_dropout,
        )

        if self.train_confidence_only:
            self.input_embedder.train()
            self.template_embedder.train()
            self.msa_module.train()
            self.pairformer_stack.train()

        return s_inputs, s, z

    def get_pairformer_output_from_s_inputs(
        self,
        input_feature_dict: dict[str, Any],
        s_inputs: torch.Tensor,
        N_cycle: int,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        mc_dropout: bool = False,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the token trunk once atom-derived ``s_inputs`` are available.

        This split is useful for high-throughput design campaigns.  Many
        sequences have the same token length but different atom counts because
        amino acids have different side-chain sizes.  Padding atom dimensions
        into one full-model batch is not a semantic no-op unless every atom
        attention and summary reduction learns about the artificial atoms.
        Instead, the runner can compute the atom input embedding per record,
        stack only the token-shaped trunk inputs here, and then finish the
        atom/diffusion/confidence tail per record.

        ``pair_mask`` is normally ``None`` because unpadded inference has no fake
        tokens.  Padded variable-token batches pass a real mask so triangular,
        MSA, template, and token-attention paths cannot read padded token keys.
        """
        z_constraint = None

        if "constraint_feature" in input_feature_dict:
            z_constraint = self.constraint_embedder(
                input_feature_dict["constraint_feature"]
            )

        s_init = self.linear_no_bias_sinit(s_inputs)  # [..., N_token, c_s]
        z_init = (
            self.linear_no_bias_zinit1(s_init)[..., None, :]
            + self.linear_no_bias_zinit2(s_init)[..., None, :, :]
        )  # [..., N_token, N_token, c_z]
        if inplace_safe:
            z_init += self.relative_position_encoding(input_feature_dict["relp"])
            z_init += self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
            if z_constraint is not None:
                z_init += z_constraint
        else:
            z_init = z_init + self.relative_position_encoding(
                input_feature_dict["relp"]
            )
            z_init = z_init + self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
            if z_constraint is not None:
                z_init = z_init + z_constraint
        # Line 6
        z = torch.zeros_like(z_init)
        s = torch.zeros_like(s_init)

        # Line 7-13 recycling
        for cycle_no in range(N_cycle):
            with torch.set_grad_enabled(
                self.training
                and (not self.train_confidence_only)
                and cycle_no == (N_cycle - 1)
            ):
                if mc_dropout:
                    z = z_init + F.dropout(
                        self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z)),
                        p=self.configs.mc_dropout_rate,
                    )
                else:
                    z = z_init + self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z))
                if inplace_safe:
                    if self.template_embedder.n_blocks > 0:
                        z += self.template_embedder(
                            input_feature_dict,
                            z,
                            pair_mask=pair_mask,
                            triangle_multiplicative=self.configs.triangle_multiplicative,
                            triangle_attention=self.configs.triangle_attention,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    z = self.msa_module(
                        input_feature_dict,
                        z,
                        s_inputs,
                        pair_mask=pair_mask,
                        triangle_multiplicative=self.configs.triangle_multiplicative,
                        triangle_attention=self.configs.triangle_attention,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
                else:
                    if self.template_embedder.n_blocks > 0:
                        z = z + self.template_embedder(
                            input_feature_dict,
                            z,
                            pair_mask=pair_mask,
                            triangle_multiplicative=self.configs.triangle_multiplicative,
                            triangle_attention=self.configs.triangle_attention,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    z = self.msa_module(
                        input_feature_dict,
                        z,
                        s_inputs,
                        pair_mask=pair_mask,
                        triangle_multiplicative=self.configs.triangle_multiplicative,
                        triangle_attention=self.configs.triangle_attention,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
                s = s_init + self.linear_no_bias_s(self.layernorm_s(s))
                s, z = self.pairformer_stack(
                    s,
                    z,
                    pair_mask=pair_mask,
                    triangle_multiplicative=self.configs.triangle_multiplicative,
                    triangle_attention=self.configs.triangle_attention,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )

        return s, z

    def sample_diffusion(self, **kwargs: Any) -> torch.Tensor:
        """
        Samples diffusion process based on the provided configurations.

        Returns:
            torch.Tensor: The result of the diffusion sampling process.
        """
        _configs = {
            key: self.configs.sample_diffusion.get(key)
            for key in [
                "gamma0",
                "gamma_min",
                "noise_scale_lambda",
                "step_scale_eta",
            ]
        }
        _configs.update(
            {
                "attn_chunk_size": (
                    self.configs.infer_setting.chunk_size if not self.training else None
                ),
                "diffusion_chunk_size": (
                    self.configs.infer_setting.sample_diffusion_chunk_size
                    if not self.training
                    else None
                ),
            }
        )
        _configs.update(
            {
                "guidance_configs": self.configs.sample_diffusion.to_dict().get(
                    "guidance"
                )
            }
        )
        return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
            sample_diffusion
        )(**_configs, **kwargs)

    def run_confidence_head(self, *args: Any, **kwargs: Any) -> Any:
        """
        Runs the confidence head with optional automatic mixed precision (AMP) disabled.

        Returns:
            Any: The output of the confidence head.
        """
        return autocasting_disable_decorator(self.configs.skip_amp.confidence_head)(
            self.confidence_head
        )(*args, **kwargs)

    def run_confidence_summary_stream(self, *args: Any, **kwargs: Any) -> Any:
        """
        Runs confidence logits in sample chunks and immediately consumes them
        into summary/full-data outputs, avoiding full-batch PAE/PDE logits.

        This is a memory optimization, not deferred confidence: all confidence
        outputs are still computed in the same inference call.  The difference
        is lifetime.  Instead of keeping every sample's large pairwise logits
        alive until summary computation, each chunk is summarized and released.
        """
        return autocasting_disable_decorator(self.configs.skip_amp.confidence_head)(
            self._run_confidence_summary_stream
        )(*args, **kwargs)

    def _run_confidence_summary_stream(
        self,
        input_feature_dict: dict[str, Any],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        pair_mask: torch.Tensor,
        x_pred_coords: torch.Tensor,
        contact_probs: torch.Tensor,
        N_recycle: int,
        mode: str,
        triangle_multiplicative: str = "torch",
        triangle_attention: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        sync_timings: bool = False,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
        sample_chunk_size = _env_int("PROTENIX_CONFIDENCE_LOGIT_CHUNK_SIZE")
        if sample_chunk_size is None:
            sample_chunk_size = _env_int("PROTENIX_SUMMARY_SAMPLE_CHUNK_SIZE") or 32
        if sample_chunk_size < 1:
            sample_chunk_size = x_pred_coords.size(-3)

        summary_confidence = []
        full_data = []
        timing = {"confidence_head": 0.0, "summary_confidence": 0.0}
        last_step = time.time()

        for (
            start,
            end,
            plddt_logits,
            pae_logits,
            pde_logits,
            _resolved_logits,
        ) in self.confidence_head.iter_logit_chunks(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_mask=pair_mask,
            x_pred_coords=x_pred_coords,
            triangle_multiplicative=triangle_multiplicative,
            triangle_attention=triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            sample_chunk_size=sample_chunk_size,
        ):
            _sync_cuda_for_timing(sync_timings)
            logits_done = time.time()
            timing["confidence_head"] += logits_done - last_step

            summary_i, full_i = autocasting_disable_decorator(True)(
                sample_confidence.compute_full_data_and_summary
            )(
                configs=self.configs,
                pae_logits=pae_logits,
                plddt_logits=plddt_logits,
                pde_logits=pde_logits,
                contact_probs=contact_probs,
                token_asym_id=input_feature_dict["asym_id"],
                token_has_frame=input_feature_dict["has_frame"],
                atom_coordinate=x_pred_coords[..., start:end, :, :],
                atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
                atom_is_polymer=1 - input_feature_dict["is_ligand"],
                N_recycle=N_recycle,
                interested_atom_mask=None,
                return_full_data=True,
                mol_id=(input_feature_dict["mol_id"] if mode != "inference" else None),
                elements_one_hot=(
                    input_feature_dict["ref_element"] if mode != "inference" else None
                ),
            )
            _sync_cuda_for_timing(sync_timings)
            summary_done = time.time()
            timing["summary_confidence"] += summary_done - logits_done

            summary_confidence.extend(summary_i)
            full_data.extend(full_i)
            del plddt_logits, pae_logits, pde_logits, _resolved_logits
            last_step = time.time()

        return summary_confidence, full_data, timing

    def main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: dict[str, Any],
        N_cycle: int,
        mode: str,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = 4,
        N_model_seed: int = 1,
        symmetric_permutation: SymmetricPermutation = None,
        mc_dropout_apply_rate: float = 0.4,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop (multiple model seeds) for the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_dict (dict[str, Any]): Label dictionary.
            N_cycle (int): Number of cycles.
            mode (str): Mode of operation (e.g., 'inference').
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to True.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to 4.
            N_model_seed (int): Number of model seeds. Defaults to 1.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object. Defaults to None.
            mc_dropout_apply_rate (float): Only for inference mode

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        # For backward compatibility, if N_model_seed > 1, process multiple seeds here
        # But in evaluation mode, this should be handled externally
        if N_model_seed > 1 and mode in ["inference"]:
            pred_dicts = []
            log_dicts = []
            time_trackers = []
            for _ in range(N_model_seed):
                pred_dict, log_dict, time_tracker = self._main_inference_loop(
                    input_feature_dict=(
                        copy.deepcopy(input_feature_dict)
                        if (N_model_seed > 1 and mode == "inference")
                        else input_feature_dict
                    ),  # the input_feature_dict is modified when mode is "inference"
                    label_dict=label_dict,
                    N_cycle=N_cycle,
                    mode=mode,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                    symmetric_permutation=symmetric_permutation,
                    mc_dropout=random.random() < mc_dropout_apply_rate,
                )
                pred_dicts.append(pred_dict)
                log_dicts.append(log_dict)
                time_trackers.append(time_tracker)

            # Combine outputs of multiple models
            def _cat(dict_list, key):
                return torch.cat([x[key] for x in dict_list], dim=0)

            def _list_join(dict_list, key):
                return sum([x[key] for x in dict_list], [])

            all_pred_dict = {
                "coordinate": _cat(pred_dicts, "coordinate"),
                "summary_confidence": _list_join(pred_dicts, "summary_confidence"),
                "full_data": _list_join(pred_dicts, "full_data"),
            }
            for key in ["plddt", "pae", "pde", "resolved"]:
                if all(key in pred_dict for pred_dict in pred_dicts):
                    all_pred_dict[key] = _cat(pred_dicts, key)

            all_log_dict = simple_merge_dict_list(log_dicts)
            all_time_dict = simple_merge_dict_list(time_trackers)
            return all_pred_dict, all_log_dict, all_time_dict
        else:
            # Single seed inference - delegate to _main_inference_loop
            return self._main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=label_dict,
                N_cycle=N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                symmetric_permutation=symmetric_permutation,
                mc_dropout=random.random() < mc_dropout_apply_rate,
            )

    def _get_dynamic_chunk_size(self, N_token: int) -> Optional[int]:
        """
        Get dynamic chunk_size based on token count

        Args:
            N_token (int): Number of tokens

        Returns:
            Optional[int]: Optimal chunk_size for the given token count
        """
        if not hasattr(self.configs.infer_setting, "chunk_size_thresholds"):
            return self.configs.infer_setting.chunk_size

        thresholds = self.configs.infer_setting.chunk_size_thresholds

        # Convert string keys to integers and sort in ascending order
        threshold_pairs = [(int(k), v) for k, v in thresholds.items()]
        sorted_thresholds = sorted(threshold_pairs, key=lambda x: x[0])

        # Find the appropriate chunk_size for the given token count
        for threshold, chunk_size in sorted_thresholds:
            if N_token <= threshold:
                return None if chunk_size == -1 else chunk_size

        # For token counts larger than the largest threshold, use smallest chunk_size
        return 32  # extreme case for very large proteins

    def _prepare_diffusion_cache(
        self,
        input_feature_dict: dict[str, Any],
        z: torch.Tensor,
    ) -> dict[str, Any]:
        """Prepare the diffusion inputs that are invariant across denoising steps."""
        cache = dict()
        if self.enable_diffusion_shared_vars_cache:
            # Lines 1-5 of Algorithm 21 calculate z in diffusion conditioning.
            cache["pair_z"] = autocasting_disable_decorator(
                self.configs.skip_amp.sample_diffusion
            )(self.diffusion_module.diffusion_conditioning.prepare_cache)(
                input_feature_dict["relp"], z, False
            )
            cache["p_lm/c_l"] = autocasting_disable_decorator(
                self.configs.skip_amp.sample_diffusion
            )(self.diffusion_module.atom_attention_encoder.prepare_cache)(
                ref_pos=input_feature_dict["ref_pos"],
                ref_charge=input_feature_dict["ref_charge"],
                ref_mask=input_feature_dict["ref_mask"],
                ref_element=input_feature_dict["ref_element"],
                ref_atom_name_chars=input_feature_dict["ref_atom_name_chars"],
                atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
                d_lm=input_feature_dict["d_lm"],
                v_lm=input_feature_dict["v_lm"],
                pad_info=input_feature_dict["pad_info"],
                r_l=True,
                z=cache["pair_z"],
                inplace_safe=False,
            )
        else:
            cache["pair_z"] = None
            cache["p_lm/c_l"] = [None, None]
        return cache

    def sample_diffusion_batch_token_transformer(
        self,
        input_feature_dicts: list[dict[str, Any]],
        s_inputs_list: list[torch.Tensor],
        s_list: list[torch.Tensor],
        z_list: list[torch.Tensor],
        chunk_size: Optional[int],
        inplace_safe: bool,
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        """Run mixed-token diffusion with one token-transformer launch per step.

        Different proteins have different token and atom counts.  The fast path
        pads both levels to the largest item in the current inference batch and
        passes masks at the attention/reduction boundaries.  Padding is
        therefore a scheduling device for the GPU, not a model input: fake
        tokens are hidden from token attention, and fake atoms are hidden from
        local atom attention and excluded from atom-to-token means.
        """
        return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
            self._sample_diffusion_batch_token_transformer
        )(
            input_feature_dicts=input_feature_dicts,
            s_inputs_list=s_inputs_list,
            s_list=s_list,
            z_list=z_list,
            chunk_size=chunk_size,
            inplace_safe=inplace_safe,
        )

    def _sample_diffusion_batch_token_transformer(
        self,
        input_feature_dicts: list[dict[str, Any]],
        s_inputs_list: list[torch.Tensor],
        s_list: list[torch.Tensor],
        z_list: list[torch.Tensor],
        chunk_size: Optional[int],
        inplace_safe: bool,
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("batched token diffusion is inference-only")

        N_sample = int(self.configs.sample_diffusion["N_sample"])

        guidance_cfg = self.configs.sample_diffusion.to_dict().get("guidance")
        if guidance_cfg is not None and guidance_cfg.get("enable", False):
            raise RuntimeError("batched token diffusion does not support guidance")

        dm = self.diffusion_module
        sync_timings = _env_flag_enabled("PROTENIX_SYNC_TIMINGS")
        token_counts = [int(s_i.shape[-2]) for s_i in s_inputs_list]
        max_tokens = max(token_counts)
        device = s_inputs_list[0].device
        dtype = s_inputs_list[0].dtype

        N_step = int(self.configs.sample_diffusion["N_step"])
        noise_schedule = self.inference_noise_scheduler(
            N_step=N_step, device=device, dtype=dtype
        )
        c_tau_last_schedule = noise_schedule[:-1]
        c_tau_schedule = noise_schedule[1:]
        gamma0 = float(self.configs.sample_diffusion["gamma0"])
        gamma_min = float(self.configs.sample_diffusion["gamma_min"])
        noise_scale_lambda = float(self.configs.sample_diffusion["noise_scale_lambda"])
        step_scale_eta = float(self.configs.sample_diffusion["step_scale_eta"])
        gamma_schedule = torch.where(
            c_tau_schedule > gamma_min,
            torch.full_like(c_tau_schedule, gamma0),
            torch.zeros_like(c_tau_schedule),
        )
        t_hat_schedule = c_tau_last_schedule * (gamma_schedule + 1)
        noise_scale_schedule = noise_scale_lambda * torch.sqrt(
            t_hat_schedule**2 - c_tau_last_schedule**2
        )

        _sync_cuda_for_timing(sync_timings)
        step_start = time.time()
        caches = [
            self._prepare_diffusion_cache(feature_dict, z_i)
            for feature_dict, z_i in zip(input_feature_dicts, z_list)
        ]
        atom_counts = [
            int(feature_dict["atom_to_token_idx"].size(-1))
            for feature_dict in input_feature_dicts
        ]
        max_atoms = max(atom_counts)
        batch_atom_transformer = (
            _env_flag_enabled_by_default("PROTENIX_BATCH_ATOM_TRANSFORMER")
            and self.enable_diffusion_shared_vars_cache
            and all(
                cache["p_lm/c_l"][0] is not None and cache["p_lm/c_l"][1] is not None
                for cache in caches
            )
        )
        batch_diffusion_conditioning = (
            _env_flag_enabled_by_default("PROTENIX_BATCH_DIFFUSION_CONDITIONING")
            and self.enable_diffusion_shared_vars_cache
            and all(cache["pair_z"] is not None for cache in caches)
        )
        conditioning_batch: Optional[dict[str, torch.Tensor]] = None
        if batch_diffusion_conditioning:
            # DiffusionConditioning's per-step work is token-local once pair_z
            # has been cached.  Stack the cached pair tensor and token singles
            # once, then run the conditioning transition for the whole batch.
            conditioning_batch = {
                "relp": _stack_padded_token_pairs(
                    [feature_dict["relp"] for feature_dict in input_feature_dicts],
                    max_tokens=max_tokens,
                    channel_first=False,
                ),
                "s_inputs": _stack_padded_token_vectors(s_inputs_list, max_tokens),
                "s_trunk": _stack_padded_token_vectors(s_list, max_tokens),
                "pair_z": _stack_padded_token_pairs(
                    [cache["pair_z"] for cache in caches],
                    max_tokens=max_tokens,
                    channel_first=False,
                ),
            }
        atom_batch: Optional[dict[str, Any]] = None
        if batch_atom_transformer:
            # Atom caches already contain the t-invariant pair conditioning.
            # Padding these cached local-attention trunks lets all proteins in
            # the batch share one H100 launch per atom-transformer block.  The
            # mask below is still required: zero padding alone would allow fake
            # atoms to become attention keys and would corrupt token means.
            max_atom_blocks = max(
                int(cache["p_lm/c_l"][0].shape[-4]) for cache in caches
            )
            atom_batch = {
                "atom_to_token_idx": _stack_padded_atom_indices(
                    [
                        feature_dict["atom_to_token_idx"]
                        for feature_dict in input_feature_dicts
                    ],
                    max_atoms=max_atoms,
                ),
                "atom_mask": _make_atom_mask(atom_counts, max_atoms, device),
                "ref_pos": _stack_padded_axis(
                    [feature_dict["ref_pos"] for feature_dict in input_feature_dicts],
                    axis=0,
                    max_size=max_atoms,
                ),
                "ref_charge": _stack_padded_axis(
                    [
                        feature_dict["ref_charge"]
                        for feature_dict in input_feature_dicts
                    ],
                    axis=0,
                    max_size=max_atoms,
                ),
                "ref_mask": _stack_padded_axis(
                    [feature_dict["ref_mask"] for feature_dict in input_feature_dicts],
                    axis=0,
                    max_size=max_atoms,
                ),
                "ref_atom_name_chars": _stack_padded_axis(
                    [
                        feature_dict["ref_atom_name_chars"]
                        for feature_dict in input_feature_dicts
                    ],
                    axis=0,
                    max_size=max_atoms,
                ),
                "ref_element": _stack_padded_axis(
                    [
                        feature_dict["ref_element"]
                        for feature_dict in input_feature_dicts
                    ],
                    axis=0,
                    max_size=max_atoms,
                ),
                "d_lm": _stack_padded_axis(
                    [feature_dict["d_lm"] for feature_dict in input_feature_dicts],
                    axis=0,
                    max_size=max_atom_blocks,
                ),
                "v_lm": _stack_padded_axis(
                    [feature_dict["v_lm"] for feature_dict in input_feature_dicts],
                    axis=0,
                    max_size=max_atom_blocks,
                ),
                "p_lm": _stack_padded_axis(
                    [cache["p_lm/c_l"][0] for cache in caches],
                    axis=-4,
                    max_size=max_atom_blocks,
                ),
                "c_l": _stack_padded_axis(
                    [cache["p_lm/c_l"][1] for cache in caches],
                    axis=0,
                    max_size=max_atoms,
                ),
                # Unused when p_lm/c_l are supplied, but the encoder signature
                # still requires it.  Keeping the real object documents that
                # this path relies on the cached-cache branch.
                "pad_info": input_feature_dicts[0]["pad_info"],
            }
        x_list = [
            noise_schedule[0]
            * torch.randn(
                (N_sample, feature_dict["atom_to_token_idx"].size(-1), 3),
                device=device,
                dtype=dtype,
            )
            for feature_dict in input_feature_dicts
        ]

        if hasattr(dm, "reset_perf_stats"):
            dm.reset_perf_stats()

        for c_tau, t_hat_scalar, noise_scale in zip(
            c_tau_schedule, t_hat_schedule, noise_scale_schedule
        ):
            a_tokens = []
            s_singles = []
            z_pairs = []
            r_noisies = []
            x_noisies = []
            t_hats = []
            decoder_inputs = []
            z_pair_padded = None

            for batch_idx, feature_dict in enumerate(input_feature_dicts):
                x_l = (
                    centre_random_augmentation(
                        x_input_coords=x_list[batch_idx], N_sample=1
                    )
                    .squeeze(dim=-3)
                    .to(dtype)
                )
                x_noisy = x_l + noise_scale * torch.randn(
                    size=x_l.shape, device=device, dtype=dtype
                )
                t_hat = t_hat_scalar.reshape(1).expand(N_sample).to(dtype)

                # EDM input scaling is cheap, but keep the same named timing
                # range as DiffusionModule.forward so profile summaries compare.
                with dm._profile_block("input_scale"):
                    r_noisy = (
                        x_noisy
                        / torch.sqrt(dm.sigma_data**2 + t_hat**2)[..., None, None]
                    )

                r_noisies.append(r_noisy)
                x_noisies.append(x_noisy)
                t_hats.append(t_hat)

            if batch_diffusion_conditioning:
                assert conditioning_batch is not None
                with dm._profile_block("conditioning"), torch.profiler.record_function(
                    "protenix/batched_diffusion_conditioning"
                ):
                    s_single_padded, z_pair_padded = dm.diffusion_conditioning(
                        # All samples in a denoising step share the same noise
                        # scalar.  Compute conditioning once per record and
                        # broadcast it later, mirroring DiffusionModule.forward.
                        t_hat_noise_level=torch.stack(
                            [t_hat[:1] for t_hat in t_hats], dim=0
                        ),
                        relp_feature=conditioning_batch["relp"],
                        s_inputs=conditioning_batch["s_inputs"],
                        s_trunk=conditioning_batch["s_trunk"],
                        z_trunk=None,
                        pair_z=conditioning_batch["pair_z"],
                        inplace_safe=inplace_safe,
                        use_conditioning=True,
                    )
                s_singles = [
                    s_single_padded[batch_idx, :, : token_counts[batch_idx]]
                    for batch_idx in range(len(input_feature_dicts))
                ]
                z_pairs = [
                    z_pair_padded[
                        batch_idx, : token_counts[batch_idx], : token_counts[batch_idx]
                    ]
                    for batch_idx in range(len(input_feature_dicts))
                ]
            else:
                for batch_idx, feature_dict in enumerate(input_feature_dicts):
                    cache = caches[batch_idx]
                    with dm._profile_block(
                        "conditioning"
                    ), torch.profiler.record_function(
                        "protenix/batched_token_diffusion_conditioning"
                    ):
                        s_single, z_pair = dm.diffusion_conditioning(
                            t_hat_noise_level=t_hats[batch_idx][:1],
                            relp_feature=feature_dict["relp"],
                            s_inputs=s_inputs_list[batch_idx],
                            s_trunk=s_list[batch_idx],
                            z_trunk=(
                                None
                                if cache["pair_z"] is not None
                                else z_list[batch_idx]
                            ),
                            pair_z=cache["pair_z"],
                            inplace_safe=inplace_safe,
                            use_conditioning=True,
                        )
                    s_singles.append(s_single)
                    z_pairs.append(z_pair)

            for batch_idx, feature_dict in enumerate(input_feature_dicts):
                if not batch_atom_transformer:
                    cache = caches[batch_idx]
                    with dm._profile_block(
                        "atom_encoder"
                    ), torch.profiler.record_function(
                        "protenix/batched_token_atom_attention_encoder"
                    ):
                        with dm._atom_attention_autocast():
                            a_token, q_skip, c_skip, p_skip = dm.atom_attention_encoder(
                                feature_dict["atom_to_token_idx"],
                                feature_dict["ref_pos"],
                                feature_dict["ref_charge"],
                                feature_dict["ref_mask"],
                                feature_dict["ref_atom_name_chars"],
                                feature_dict["ref_element"],
                                feature_dict["d_lm"],
                                feature_dict["v_lm"],
                                feature_dict["pad_info"],
                                r_l=r_noisies[batch_idx],
                                s=s_list[batch_idx].unsqueeze(dim=-3),
                                z=z_pairs[batch_idx].unsqueeze(dim=-4),
                                p_lm=cache["p_lm/c_l"][0],
                                c_l=cache["p_lm/c_l"][1],
                                inplace_safe=inplace_safe,
                                chunk_size=chunk_size,
                            )

                    transformer_dtype = dm._diffusion_core_dtype(a_token.dtype)
                    a_token = a_token.to(dtype=transformer_dtype)
                    if inplace_safe:
                        a_token += dm.linear_no_bias_s(
                            dm.layernorm_s(s_singles[batch_idx])
                        )
                    else:
                        a_token = a_token + dm.linear_no_bias_s(
                            dm.layernorm_s(s_singles[batch_idx])
                        )
                    a_tokens.append(a_token)
                    decoder_inputs.append(
                        (
                            batch_idx,
                            x_noisies[batch_idx],
                            t_hats[batch_idx],
                            q_skip,
                            c_skip,
                            p_skip,
                        )
                    )

            if batch_atom_transformer:
                assert atom_batch is not None
                with dm._profile_block("atom_encoder"), torch.profiler.record_function(
                    "protenix/batched_atom_attention_encoder"
                ):
                    with dm._atom_attention_autocast():
                        a_token_batch, q_skip_batch, c_skip_batch, p_skip_batch = (
                            dm.atom_attention_encoder(
                                atom_batch["atom_to_token_idx"],
                                atom_batch["ref_pos"],
                                atom_batch["ref_charge"],
                                atom_batch["ref_mask"],
                                atom_batch["ref_atom_name_chars"],
                                atom_batch["ref_element"],
                                atom_batch["d_lm"],
                                atom_batch["v_lm"],
                                atom_batch["pad_info"],
                                r_l=_stack_padded_axis(
                                    r_noisies, axis=-2, max_size=max_atoms
                                ),
                                s=(
                                    conditioning_batch["s_trunk"].unsqueeze(dim=-3)
                                    if conditioning_batch is not None
                                    else _stack_padded_axis(
                                        [s_i.unsqueeze(dim=-3) for s_i in s_list],
                                        axis=-2,
                                        max_size=max_tokens,
                                    )
                                ),
                                z=(
                                    z_pair_padded
                                    if z_pair_padded is not None
                                    else _stack_padded_token_pairs(
                                        z_pairs,
                                        max_tokens=max_tokens,
                                        channel_first=False,
                                    )
                                ).unsqueeze(dim=-4),
                                p_lm=atom_batch["p_lm"],
                                c_l=atom_batch["c_l"],
                                inplace_safe=inplace_safe,
                                chunk_size=chunk_size,
                                atom_mask=atom_batch["atom_mask"],
                            )
                        )

                transformer_dtype = dm._diffusion_core_dtype(a_token_batch.dtype)
                a_token_batch = a_token_batch.to(dtype=transformer_dtype)
                s_single_batch = _stack_padded_axis(
                    s_singles, axis=-2, max_size=max_tokens
                )
                if inplace_safe:
                    a_token_batch += dm.linear_no_bias_s(dm.layernorm_s(s_single_batch))
                else:
                    a_token_batch = a_token_batch + dm.linear_no_bias_s(
                        dm.layernorm_s(s_single_batch)
                    )
                a_batch = a_token_batch
                s_batch = s_single_batch.to(dtype=transformer_dtype)
            else:
                transformer_dtype = dm._diffusion_core_dtype(a_tokens[0].dtype)
                s_batch = _stack_padded_axis(
                    [s_i.to(dtype=transformer_dtype) for s_i in s_singles],
                    axis=-2,
                    max_size=max_tokens,
                )
                a_batch = _stack_padded_axis(
                    a_tokens,
                    axis=-2,
                    max_size=max_tokens,
                )

            token_mask = _make_token_mask(token_counts, max_tokens, device)
            use_sample_axis_transformer = (
                N_sample > 1 and _diffusion_transformer_sample_axis_enabled()
            )
            if z_pair_padded is not None:
                # Batched conditioning has already produced one padded
                # [B, N, N, C] pair tensor.  Reusing it avoids a redundant
                # slice/list/restack cycle on every diffusion step.
                if self.enable_efficient_fusion:
                    z_batch = permute_final_dims(
                        dm.normalize(z_pair_padded.to(dtype=transformer_dtype)),
                        [2, 0, 1],
                    ).contiguous()
                else:
                    z_batch = z_pair_padded.to(dtype=transformer_dtype)
            else:
                if self.enable_efficient_fusion:
                    z_pairs_for_transformer = [
                        permute_final_dims(
                            dm.normalize(z_pair.to(dtype=transformer_dtype)),
                            [2, 0, 1],
                        ).contiguous()
                        for z_pair in z_pairs
                    ]
                else:
                    z_pairs_for_transformer = [
                        z_pair.to(dtype=transformer_dtype) for z_pair in z_pairs
                    ]
                z_batch = _stack_padded_token_pairs(
                    z_pairs_for_transformer,
                    max_tokens=max_tokens,
                    channel_first=self.enable_efficient_fusion,
                )

            if use_sample_axis_transformer:
                # Experimental low-sample path: keep the sample axis explicit so
                # pair bias can stay [record, 1, head, token, token].  Flattening
                # to [record * sample, ...] is robust and still the default, but
                # it materializes N_sample copies of the same pair/padding bias.
                # The transformer code broadcasts the singleton axis and relies
                # on PROTENIX_RANK5_FULL_ATTENTION_BF16=1 to keep full token
                # attention on tensor-core BF16 SDPA instead of the generic
                # rank-5 FP32 safeguard used for atom local attention.
                a_transformer = _expand_sample_axis(a_batch, N_sample)
                # ``s_batch`` already carries a singleton sample axis from
                # DiffusionConditioning: [record, 1, token, channel].  Adding
                # another ``None`` here would produce
                # [record, 1, 1, token, channel].  PyTorch broadcasting would
                # then introduce a spurious extra record axis inside AdaLN, and
                # SDPA would see the sample dimension where the head dimension
                # should be.
                s_transformer = s_batch
                token_mask_transformer = token_mask[:, None, :]
            else:
                a_transformer = _flatten_record_sample_axes(a_batch, N_sample)
                s_transformer = _flatten_record_sample_axes(s_batch, N_sample)
                token_mask_transformer = _flatten_record_sample_mask(
                    token_mask, N_sample
                )

            with dm._profile_block("transformer"), torch.profiler.record_function(
                "protenix/batched_token_diffusion_transformer"
            ):
                with dm._diffusion_core_autocast(transformer_dtype):
                    a_transformer = dm.diffusion_transformer(
                        a=a_transformer,
                        s=s_transformer,
                        # ``z`` is identical for every diffusion sample of a
                        # record.  Keep it at record grain so each block
                        # projects pair bias once per protein, then repeats
                        # the smaller head bias over flattened sample lanes.
                        z=z_batch,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                        enable_efficient_fusion=self.enable_efficient_fusion,
                        token_mask=token_mask_transformer,
                        z_sample_count=N_sample,
                        z_sample_axis=use_sample_axis_transformer,
                    )
            if a_transformer.dtype != torch.float32:
                a_transformer = a_transformer.to(dtype=torch.float32)
            if use_sample_axis_transformer:
                a_batch = a_transformer
            else:
                a_batch = a_transformer.reshape(
                    len(input_feature_dicts),
                    N_sample,
                    max_tokens,
                    a_transformer.shape[-1],
                )

            if batch_atom_transformer:
                assert atom_batch is not None
                a_decode_batch = dm.layernorm_a(a_batch)
                with dm._profile_block("atom_decoder"), torch.profiler.record_function(
                    "protenix/batched_atom_attention_decoder"
                ):
                    with dm._atom_attention_autocast():
                        r_update_batch = dm.atom_attention_decoder(
                            atom_to_token_idx=atom_batch["atom_to_token_idx"],
                            a=a_decode_batch,
                            q_skip=q_skip_batch,
                            c_skip=c_skip_batch,
                            p_skip=p_skip_batch,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                            atom_mask=atom_batch["atom_mask"],
                        )
                decoder_outputs = [
                    (
                        batch_idx,
                        x_noisies[batch_idx],
                        t_hats[batch_idx],
                        r_update_batch[batch_idx, :, : atom_counts[batch_idx]],
                    )
                    for batch_idx in range(len(input_feature_dicts))
                ]
            else:
                decoder_outputs = []
                for (
                    batch_idx,
                    x_noisy,
                    t_hat,
                    q_skip,
                    c_skip,
                    p_skip,
                ) in decoder_inputs:
                    n_token = token_counts[batch_idx]
                    a_token = dm.layernorm_a(
                        a_batch[batch_idx, :, :n_token]
                    )

                    with dm._profile_block(
                        "atom_decoder"
                    ), torch.profiler.record_function(
                        "protenix/batched_token_atom_attention_decoder"
                    ):
                        with dm._atom_attention_autocast():
                            r_update = dm.atom_attention_decoder(
                                atom_to_token_idx=input_feature_dicts[batch_idx][
                                    "atom_to_token_idx"
                                ],
                                a=a_token,
                                q_skip=q_skip,
                                c_skip=c_skip,
                                p_skip=p_skip,
                                inplace_safe=inplace_safe,
                                chunk_size=chunk_size,
                            )
                    decoder_outputs.append((batch_idx, x_noisy, t_hat, r_update))

            for batch_idx, x_noisy, t_hat, r_update in decoder_outputs:

                with dm._profile_block("output_rescale"):
                    s_ratio = (t_hat / dm.sigma_data)[..., None, None].to(
                        r_update.dtype
                    )
                    x_denoised = (
                        1 / (1 + s_ratio**2) * x_noisy
                        + t_hat[..., None, None]
                        / torch.sqrt(1 + s_ratio**2)
                        * r_update
                    ).to(r_update.dtype)

                delta = (x_noisy - x_denoised) / t_hat[..., None, None]
                dt = c_tau - t_hat
                x_list[batch_idx] = (
                    x_noisy + step_scale_eta * dt[..., None, None] * delta
                )

        _sync_cuda_for_timing(sync_timings)
        time_tracker: dict[str, Any] = {"diffusion": time.time() - step_start}
        time_tracker["diffusion_atom_transformer_batched"] = batch_atom_transformer
        time_tracker["diffusion_conditioning_batched"] = batch_diffusion_conditioning
        if hasattr(dm, "consume_perf_stats"):
            time_tracker.update(dm.consume_perf_stats())
        return x_list, time_tracker

    def finish_inference_from_pairformer(
        self,
        input_feature_dict: dict[str, Any],
        s_inputs: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        label_dict: Optional[dict[str, Any]],
        N_cycle: int,
        mode: str,
        inplace_safe: bool,
        chunk_size: Optional[int],
        symmetric_permutation: SymmetricPermutation = None,
        step_st: Optional[float] = None,
        pairformer_sec: Optional[float] = None,
        precomputed_coordinate: Optional[torch.Tensor] = None,
        precomputed_diffusion_time: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """Finish inference after the token trunk has produced ``s`` and ``z``.

        The normal full-model path calls this immediately after
        ``get_pairformer_output``.  The campaign batching path can instead
        compute atom input embeddings for several same-token records, run this
        trunk in one batched call, then call this tail once per record.  Keeping
        this boundary explicit avoids padding variable atom counts through atom
        attention, diffusion, and confidence reductions.
        """
        sync_timings = _env_flag_enabled("PROTENIX_SYNC_TIMINGS")
        pred_dict: dict[str, Any] = {}
        log_dict: dict[str, Any] = {}
        time_tracker: dict[str, Any] = {}
        if step_st is None:
            step_st = time.time()

        keys_to_delete = []
        for key in input_feature_dict.keys():
            if "template_" in key or key in [
                "msa",
                "has_deletion",
                "deletion_value",
                "profile",
                "deletion_mean",
                # "token_bonds",
            ]:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del input_feature_dict[key]
        _sync_cuda_for_timing(sync_timings)
        step_trunk = time.time()
        pairformer_time = (
            step_trunk - step_st if pairformer_sec is None else pairformer_sec
        )
        time_tracker.update({"pairformer": pairformer_time})
        if precomputed_coordinate is None:
            # Sample diffusion
            # [..., N_sample, N_atom, 3]
            N_sample = self.configs.sample_diffusion["N_sample"]
            N_step = self.configs.sample_diffusion["N_step"]

            noise_schedule = self.inference_noise_scheduler(
                N_step=N_step, device=s_inputs.device, dtype=s_inputs.dtype
            )
            cache = self._prepare_diffusion_cache(input_feature_dict, z)
            if hasattr(self.diffusion_module, "reset_perf_stats"):
                self.diffusion_module.reset_perf_stats()
            pred_dict["coordinate"] = self.sample_diffusion(
                denoise_net=self.diffusion_module,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s,
                z_trunk=None if cache["pair_z"] is not None else z,
                pair_z=cache["pair_z"],
                p_lm=cache["p_lm/c_l"][0],
                c_l=cache["p_lm/c_l"][1],
                N_sample=N_sample,
                noise_schedule=noise_schedule,
                inplace_safe=inplace_safe,
                enable_efficient_fusion=self.enable_efficient_fusion,
            )
        else:
            pred_dict["coordinate"] = precomputed_coordinate

        _sync_cuda_for_timing(sync_timings)
        step_diffusion = time.time()
        if precomputed_diffusion_time is None:
            time_tracker.update({"diffusion": step_diffusion - step_trunk})
        else:
            time_tracker.update(precomputed_diffusion_time)
        if (
            precomputed_diffusion_time is None
            and hasattr(self.diffusion_module, "consume_perf_stats")
        ):
            time_tracker.update(self.diffusion_module.consume_perf_stats())

        # Distogram logits: log contact_probs only, to reduce the dimension
        pred_dict["contact_probs"] = autocasting_disable_decorator(True)(
            sample_confidence.compute_contact_prob
        )(
            distogram_logits=self.distogram_head(z),
            **sample_confidence.get_bin_params(self.configs.loss.distogram),
        )  # [N_token, N_token]
        _sync_cuda_for_timing(sync_timings)
        step_contact_probs = time.time()
        time_tracker.update({"contact_probs": step_contact_probs - step_diffusion})

        stream_confidence_summary = (
            _env_flag_enabled("PROTENIX_STREAM_CONFIDENCE_SUMMARY")
            and not self.training
            and mode == "inference"
            and label_dict is None
            and symmetric_permutation is None
        )
        if stream_confidence_summary:
            (
                pred_dict["summary_confidence"],
                pred_dict["full_data"],
                stream_timing,
            ) = self.run_confidence_summary_stream(
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s,
                z_trunk=z,
                pair_mask=None,
                x_pred_coords=pred_dict["coordinate"],
                contact_probs=pred_dict["contact_probs"],
                N_recycle=N_cycle,
                mode=mode,
                triangle_multiplicative=self.configs.triangle_multiplicative,
                triangle_attention=self.configs.triangle_attention,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                sync_timings=sync_timings,
            )

            _sync_cuda_for_timing(sync_timings)
            step_summary = time.time()
            confidence_head_time = stream_timing["confidence_head"]
            summary_time = stream_timing["summary_confidence"]
            contact_time = time_tracker["contact_probs"]
            model_forward_time = step_contact_probs - step_st + confidence_head_time
            model_forward_with_summary = step_summary - step_st
            if precomputed_diffusion_time is not None:
                model_forward_time = (
                    time_tracker["pairformer"]
                    + time_tracker["diffusion"]
                    + contact_time
                    + confidence_head_time
                )
                model_forward_with_summary = model_forward_time + summary_time
            time_tracker.update(
                {
                    "confidence_head": confidence_head_time,
                    "confidence": contact_time + confidence_head_time,
                    "model_forward": model_forward_time,
                    "summary_confidence": summary_time,
                    "model_forward_with_summary": model_forward_with_summary,
                    "confidence_summary_stream": True,
                }
            )
            return pred_dict, log_dict, time_tracker

        # Confidence logits
        (
            pred_dict["plddt"],
            pred_dict["pae"],
            pred_dict["pde"],
            pred_dict["resolved"],
        ) = self.run_confidence_head(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            pair_mask=None,
            x_pred_coords=pred_dict["coordinate"],
            triangle_multiplicative=self.configs.triangle_multiplicative,
            triangle_attention=self.configs.triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        _sync_cuda_for_timing(sync_timings)
        step_confidence = time.time()
        confidence_head_time = step_confidence - step_contact_probs
        confidence_time = step_confidence - step_diffusion
        model_forward_time = step_confidence - step_st
        if precomputed_diffusion_time is not None:
            confidence_time = time_tracker["contact_probs"] + confidence_head_time
            model_forward_time = (
                time_tracker["pairformer"] + time_tracker["diffusion"] + confidence_time
            )
        time_tracker.update(
            {
                "confidence_head": confidence_head_time,
                "confidence": confidence_time,
                "model_forward": model_forward_time,
            }
        )

        # Permutation: when label is given, permute coordinates and other heads
        if label_dict is not None and symmetric_permutation is not None:
            pred_dict, log_dict = symmetric_permutation.permute_inference_pred_dict(
                input_feature_dict=input_feature_dict,
                pred_dict=pred_dict,
                label_dict=label_dict,
                permute_by_pocket=("pocket_mask" in label_dict)
                and ("interested_ligand_mask" in label_dict),
            )
            last_step_seconds = step_confidence
            _sync_cuda_for_timing(sync_timings)
            time_tracker.update({"permutation": time.time() - last_step_seconds})
        _sync_cuda_for_timing(sync_timings)
        step_before_summary = time.time()

        # Summary Confidence & Full Data
        # Computed after coordinates and logits are permuted
        if label_dict is None:
            interested_atom_mask = None
        else:
            interested_atom_mask = label_dict.get("interested_ligand_mask", None)
        (
            pred_dict["summary_confidence"],
            pred_dict["full_data"],
        ) = autocasting_disable_decorator(True)(
            sample_confidence.compute_full_data_and_summary
        )(
            configs=self.configs,
            pae_logits=pred_dict["pae"],
            plddt_logits=pred_dict["plddt"],
            pde_logits=pred_dict["pde"],
            contact_probs=pred_dict.get(
                "per_sample_contact_probs", pred_dict["contact_probs"]
            ),
            token_asym_id=input_feature_dict["asym_id"],
            token_has_frame=input_feature_dict["has_frame"],
            atom_coordinate=pred_dict["coordinate"],
            atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
            atom_is_polymer=1 - input_feature_dict["is_ligand"],
            N_recycle=N_cycle,
            interested_atom_mask=interested_atom_mask,
            return_full_data=True,
            mol_id=(input_feature_dict["mol_id"] if mode != "inference" else None),
            elements_one_hot=(
                input_feature_dict["ref_element"] if mode != "inference" else None
            ),
        )
        _sync_cuda_for_timing(sync_timings)
        step_summary = time.time()
        summary_confidence_time = step_summary - step_before_summary
        model_forward_with_summary = step_summary - step_st
        if precomputed_diffusion_time is not None:
            model_forward_with_summary = (
                time_tracker["model_forward"] + summary_confidence_time
            )
        time_tracker.update(
            {
                "summary_confidence": summary_confidence_time,
                "model_forward_with_summary": model_forward_with_summary,
            }
        )

        return pred_dict, log_dict, time_tracker

    def _main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: dict[str, Any],
        N_cycle: int,
        mode: str,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = 4,
        symmetric_permutation: SymmetricPermutation = None,
        mc_dropout: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop (single model seed) for the Alphafold3 model.
        mc_dropout: do not use by default

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        sync_timings = _env_flag_enabled("PROTENIX_SYNC_TIMINGS")
        _sync_cuda_for_timing(sync_timings)
        step_st = time.time()
        N_token = input_feature_dict["residue_index"].shape[-1]

        # Apply dynamic chunk_size if enabled (otherwise keep the passed chunk_size)
        if (
            hasattr(self.configs.infer_setting, "dynamic_chunk_size")
            and self.configs.infer_setting.dynamic_chunk_size
        ):
            chunk_size = self._get_dynamic_chunk_size(N_token)
        # If dynamic chunking is disabled, chunk_size keeps its original value from the function parameter

        s_inputs, s, z = self.get_pairformer_output(
            input_feature_dict=input_feature_dict,
            N_cycle=N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            mc_dropout=mc_dropout,
        )

        return self.finish_inference_from_pairformer(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s=s,
            z=z,
            label_dict=label_dict,
            N_cycle=N_cycle,
            mode=mode,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            symmetric_permutation=symmetric_permutation,
            step_st=step_st,
        )

    def main_train_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: dict[str, Any],
        label_dict: dict[str, Any],
        N_cycle: int,
        symmetric_permutation: SymmetricPermutation,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main training loop for the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_full_dict (dict[str, Any]): Full label dictionary (uncropped).
            label_dict (dict): Label dictionary (cropped).
            N_cycle (int): Number of cycles.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object.
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.
        """

        s_inputs, s, z = self.get_pairformer_output(
            input_feature_dict=input_feature_dict,
            N_cycle=N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        log_dict = {}
        pred_dict = {}

        cache = dict()
        if self.enable_diffusion_shared_vars_cache:
            cache["pair_z"] = autocasting_disable_decorator(
                self.configs.skip_amp.sample_diffusion
            )(self.diffusion_module.diffusion_conditioning.prepare_cache)(
                input_feature_dict["relp"], z, False
            )
            cache["p_lm/c_l"] = autocasting_disable_decorator(
                self.configs.skip_amp.sample_diffusion
            )(self.diffusion_module.atom_attention_encoder.prepare_cache)(
                ref_pos=input_feature_dict["ref_pos"],
                ref_charge=input_feature_dict["ref_charge"],
                ref_mask=input_feature_dict["ref_mask"],
                ref_element=input_feature_dict["ref_element"],
                ref_atom_name_chars=input_feature_dict["ref_atom_name_chars"],
                atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
                d_lm=input_feature_dict["d_lm"],
                v_lm=input_feature_dict["v_lm"],
                pad_info=input_feature_dict["pad_info"],
                r_l=True,
                z=cache["pair_z"],
                inplace_safe=False,
            )
        else:
            cache["pair_z"] = None
            cache["p_lm/c_l"] = [None, None]
        # Mini-rollout: used for confidence and label permutation
        with torch.no_grad():
            # [..., 1, N_atom, 3]
            N_sample_mini_rollout = self.configs.sample_diffusion[
                "N_sample_mini_rollout"
            ]  # =1
            N_step_mini_rollout = self.configs.sample_diffusion["N_step_mini_rollout"]
            self.diffusion_module.eval()  # use eval mode for mini-rollout
            coordinate_mini = self.sample_diffusion(
                denoise_net=self.diffusion_module,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs.detach(),
                s_trunk=s.detach(),
                z_trunk=None if cache["pair_z"] is not None else z.detach(),
                pair_z=None if cache["pair_z"] is None else cache["pair_z"].detach(),
                p_lm=(
                    None
                    if cache["p_lm/c_l"][0] is None
                    else cache["p_lm/c_l"][0].detach()
                ),
                c_l=(
                    None
                    if cache["p_lm/c_l"][1] is None
                    else cache["p_lm/c_l"][1].detach()
                ),
                N_sample=N_sample_mini_rollout,
                noise_schedule=self.inference_noise_scheduler(
                    N_step=N_step_mini_rollout,
                    device=s_inputs.device,
                    dtype=s_inputs.dtype,
                ),
                enable_efficient_fusion=self.enable_efficient_fusion,
            )
            self.diffusion_module.train()
            coordinate_mini.detach_()
            pred_dict["coordinate_mini"] = coordinate_mini

            # Permute ground truth to match mini-rollout prediction
            (
                label_dict,
                perm_log_dict,
            ) = symmetric_permutation.permute_label_to_match_mini_rollout(
                coordinate_mini,
                input_feature_dict,
                label_dict,
                label_full_dict,
            )
            log_dict.update(perm_log_dict)

        # Confidence: use mini-rollout prediction, and detach token embeddings
        drop_embedding = (
            random.random() < self.configs.model.confidence_embedding_drop_rate
        )
        plddt_pred, pae_pred, pde_pred, resolved_pred = self.run_confidence_head(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            pair_mask=None,
            x_pred_coords=coordinate_mini,
            use_embedding=not drop_embedding,
            triangle_multiplicative=self.configs.triangle_multiplicative,
            triangle_attention=self.configs.triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        pred_dict.update(
            {
                "plddt": plddt_pred,
                "pae": pae_pred,
                "pde": pde_pred,
                "resolved": resolved_pred,
            }
        )

        if self.train_confidence_only:
            # Skip diffusion loss and distogram loss. Return now.
            return pred_dict, label_dict, log_dict

        # Denoising: use permuted coords to generate noisy samples and perform denoising
        # x_denoised: [..., N_sample, N_atom, 3]
        # x_noise_level: [..., N_sample]
        N_sample = self.diffusion_batch_size
        drop_conditioning = (
            random.random() < self.configs.model.condition_embedding_drop_rate
        )
        _, x_denoised, x_noise_level = autocasting_disable_decorator(
            self.configs.skip_amp.sample_diffusion_training
        )(sample_diffusion_training)(
            noise_sampler=self.train_noise_sampler,
            denoise_net=self.diffusion_module,
            label_dict=label_dict,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=None if cache["pair_z"] is not None else z,
            pair_z=cache["pair_z"],
            p_lm=cache["p_lm/c_l"][0],
            c_l=cache["p_lm/c_l"][1],
            N_sample=N_sample,
            diffusion_chunk_size=self.configs.diffusion_chunk_size,
            use_conditioning=not drop_conditioning,
            enable_efficient_fusion=self.enable_efficient_fusion,
        )
        pred_dict.update(
            {
                "distogram": autocasting_disable_decorator(True)(self.distogram_head)(
                    z
                ),
                # [..., N_sample=48, N_atom, 3]: diffusion loss
                "coordinate": x_denoised,
                "noise_level": x_noise_level,
            }
        )

        # Permute symmetric atom/chain in each sample to match true structure
        # Note: currently chains cannot be permuted since label is cropped
        (
            pred_dict,
            perm_log_dict,
            _,
            _,
        ) = symmetric_permutation.permute_diffusion_sample_to_match_label(
            input_feature_dict, pred_dict, label_dict, stage="train"
        )
        log_dict.update(perm_log_dict)
        log_dict.update({"noise_level": x_noise_level})

        return pred_dict, label_dict, log_dict

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: dict[str, Any],
        label_dict: dict[str, Any],
        mode: str = "inference",
        current_step: Optional[int] = None,
        symmetric_permutation: SymmetricPermutation = None,
        disable_inplace: bool = False,
        mc_dropout_apply_rate: float = 0.4,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Forward pass of the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_full_dict (dict[str, Any]): Full label dictionary (uncropped).
            label_dict (dict[str, Any]): Label dictionary (cropped).
            mode (str): Mode of operation ('train', 'inference', 'eval'). Defaults to 'inference'.
            current_step (Optional[int]): Current training step. Defaults to None.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.
        """

        assert mode in ["train", "eval", "inference"]
        not_use_gradient = not (self.training or torch.is_grad_enabled())
        inplace_safe = not_use_gradient and (not disable_inplace)

        input_feature_dict = self.relative_position_encoding.generate_relp(
            input_feature_dict
        )
        input_feature_dict = update_input_feature_dict(input_feature_dict)

        if mode == "train":
            nc_rng = np.random.RandomState(current_step)
            N_cycle = nc_rng.randint(1, self.N_cycle + 1)
            assert self.training
            assert label_dict is not None
            assert symmetric_permutation is not None

            pred_dict, label_dict, log_dict = self.main_train_loop(
                input_feature_dict=input_feature_dict,
                label_full_dict=label_full_dict,
                label_dict=label_dict,
                N_cycle=N_cycle,
                symmetric_permutation=symmetric_permutation,
                inplace_safe=inplace_safe,
                chunk_size=None,
            )
            log_dict["N_cycle"] = N_cycle
        elif mode == "inference":
            pred_dict, log_dict, time_tracker = self.main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=None,
                N_cycle=self.N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=self.configs.infer_setting.chunk_size,
                N_model_seed=self.N_model_seed,
                symmetric_permutation=None,
                mc_dropout_apply_rate=mc_dropout_apply_rate,
            )
            log_dict.update({"time": time_tracker})
        elif mode == "eval":
            if label_dict is not None:
                assert (
                    label_dict["coordinate"].size()
                    == label_full_dict["coordinate"].size()
                )
                label_dict.update(label_full_dict)

            pred_dict, log_dict, time_tracker = self.main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=label_dict,
                N_cycle=self.N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=self.configs.infer_setting.chunk_size,
                N_model_seed=1,
                symmetric_permutation=symmetric_permutation,
                mc_dropout_apply_rate=mc_dropout_apply_rate,
            )
            log_dict.update({"time": time_tracker})

        return pred_dict, label_dict, log_dict
