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
import json
import logging
import os
import time
import traceback
import urllib.request
from argparse import Namespace
from collections.abc import Mapping as MappingABC
from contextlib import nullcontext
from numbers import Number
from os.path import exists as opexists, join as opjoin
from typing import Any, Mapping

import torch
import torch.distributed as dist

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs
from protenix.config.config import parse_configs, parse_sys_args
from protenix.data.inference.infer_dataloader import get_inference_dataloader
from protenix.model.protenix import Protenix, update_input_feature_dict
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.seed import seed_everything
from protenix.utils.torch_utils import to_device
from protenix.web_service.dependency_url import URL

from runner.dumper import DataDumper

logger = logging.getLogger(__name__)
"""
Due to the fair-esm repository being archived,
it can no longer be updated to support newer versions of PyTorch.
Starting from PyTorch 2.6, the default value of the weights_only argument
in torch.load has been changed from False to True,
which enhances security but causes loading ESM models to fail
with the following error:

_pickle.UnpicklingError: Weights only load failed. This file can still be loaded...
This error occurs because the model file contains argparse.Namespace,
which is not allowed by default in the secure unpickling process of PyTorch 2.6+.

✅ Solution (Patch)
Since we cannot modify the fair-esm source code,
we can apply a patch before calling load_model_and_alphabet_local
by manually adding argparse.Namespace to PyTorch's safe globals list.
"""

torch.serialization.add_safe_globals([Namespace])


class InferenceRunner(object):
    """
    Runner class for AlphaFold3 model inference.
    Handles environment setup, model initialization, and running predictions.

    Args:
        configs (Any): Configuration object for inference.
    """

    def __init__(self, configs: Any) -> None:
        self.configs = configs
        self.last_log_dict = {}
        self.last_batch_time_summary = None
        self.init_env()
        self.init_basics()
        self.init_model()
        self.load_checkpoint()
        self.init_dumper(
            need_atom_confidence=configs.need_atom_confidence,
            sorted_by_ranking_score=configs.sorted_by_ranking_score,
        )

    def init_env(self) -> None:
        """
        Initialize the execution environment, including CUDA and distributed setup.
        """
        self.print(
            f"Distributed environment: world size: {DIST_WRAPPER.world_size}, "
            f"global rank: {DIST_WRAPPER.rank}, local rank: {DIST_WRAPPER.local_rank}"
        )
        self.use_cuda = torch.cuda.device_count() > 0
        if self.use_cuda:
            self.device = torch.device(f"cuda:{DIST_WRAPPER.local_rank}")
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
            devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
            logging.info(
                f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]"
            )
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if DIST_WRAPPER.world_size > 1:
            dist.init_process_group(backend="nccl")

        if self.configs.triangle_attention == "deepspeed":
            env = os.getenv("CUTLASS_PATH", None)
            self.print(f"env: {env}")
            assert env is not None, (
                "If use deepspeed (ds4sci), set CUTLASS_PATH environment variable "
                "per instructions at "
                "https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/"
            )
            logging.info(
                "Kernels will be compiled when DS4Sci_EvoformerAttention "
                "is first called."
            )

        use_fastlayernorm = os.getenv("LAYERNORM_TYPE", "fast_layernorm")
        if use_fastlayernorm == "fast_layernorm":
            logging.info(
                "Kernels will be compiled when fast_layernorm is first called."
            )

        logging.info("Finished environment initialization.")

    def init_basics(self) -> None:
        """
        Initialize basic directory structures for dumping results and errors.
        """
        self.dump_dir = self.configs.dump_dir
        self.error_dir = opjoin(self.dump_dir, "ERR")
        os.makedirs(self.dump_dir, exist_ok=True)
        os.makedirs(self.error_dir, exist_ok=True)

    def init_model(self) -> None:
        """
        Initialize the Protenix model and move it to the appropriate device.
        """
        self.model = Protenix(self.configs).to(self.device)

    def load_checkpoint(self) -> None:
        """
        Load model weights from a checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint path does not exist.
        """
        checkpoint_path = opjoin(
            self.configs.load_checkpoint_dir, f"{self.configs.model_name}.pt"
        )
        if not opexists(checkpoint_path):
            raise FileNotFoundError(
                f"Given checkpoint path not exist [{checkpoint_path}]"
            )

        self.print(
            f"Loading from {checkpoint_path}, strict: {self.configs.load_strict}"
        )
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        sample_key = list(checkpoint["model"].keys())[0]
        self.print(f"Sampled key: {sample_key}")
        if sample_key.startswith("module."):  # DDP checkpoint has module. prefix
            checkpoint["model"] = {
                k[len("module.") :]: v for k, v in checkpoint["model"].items()
            }
        self.model.load_state_dict(
            state_dict=checkpoint["model"],
            strict=self.configs.load_strict,
        )
        self.model.eval()
        self.print("Finish loading checkpoint.")

        def count_parameters(model: torch.nn.Module) -> float:
            """Count total parameters in millions."""
            total_params = sum(p.numel() for p in model.parameters())
            return total_params / 1e6

        self.print(f"Model parameters: {count_parameters(self.model):.2f}M")

    def init_dumper(
        self, need_atom_confidence: bool = False, sorted_by_ranking_score: bool = True
    ) -> None:
        """
        Initialize the data dumper for saving predictions.

        Args:
            need_atom_confidence (bool): Whether to dump atom-level confidence.
            sorted_by_ranking_score (bool): Whether to sort results by ranking score.
        """
        self.dumper = DataDumper(
            base_dir=self.dump_dir,
            need_atom_confidence=need_atom_confidence,
            sorted_by_ranking_score=sorted_by_ranking_score,
        )

    # Adapted from runner.train.AF3Trainer.evaluate
    @torch.no_grad()
    def predict(self, data: Mapping[str, Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Run model prediction on the provided data.

        Args:
            data (Mapping[str, Mapping[str, Any]]): Input data dictionary.

        Returns:
            dict[str, torch.Tensor]: Prediction results.
        """
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]

        enable_amp = (
            torch.autocast(device_type="cuda", dtype=eval_precision)
            if torch.cuda.is_available()
            else nullcontext()
        )

        self.last_log_dict = {}
        self.last_batch_time_summary = None
        data = to_device(data, self.device)
        with enable_amp:
            prediction, _, log_dict = self.model(
                input_feature_dict=data["input_feature_dict"],
                label_full_dict=None,
                label_dict=None,
                mode="inference",
                mc_dropout_apply_rate=self.configs.mc_dropout_apply_rate,
            )
        self.last_log_dict = log_dict

        return prediction

    @torch.no_grad()
    def predict_token_batch(
        self,
        data_items: list[Mapping[str, Any]],
        *,
        exact_token_trunk: bool = False,
    ) -> list[dict[str, torch.Tensor]]:
        """Batch the expensive token trunk for ragged campaign records.

        Protein design campaigns commonly mutate residues while keeping the
        target length fixed.  Those records share the pairformer trunk shape but
        do not share atom shapes because residue side chains have different atom
        counts.  Padding fake atoms through atom attention is risky: fake keys
        change softmax denominators and fake atoms also affect atom-level
        confidence reductions unless every downstream operation is mask-aware.

        This path therefore keeps atom-shaped work ragged at the input boundary
        by running the atom input embedding per record.  By default, different
        token counts are padded only through the token trunk with a real
        ``pair_mask`` and then cropped back before the atom-shaped tail.  When
        ``exact_token_trunk`` is true, the pairformer trunk itself also runs
        per record at the physical token length.  That protects the largest
        schedule-sensitive reduction boundary while still allowing the cheaper
        diffusion tail to use its own masked batching path.
        """
        if len(data_items) == 1:
            return [
                self.predict(
                    {"input_feature_dict": data_items[0]["input_feature_dict"]}
                )
            ]

        self.last_log_dict = {}
        self.last_batch_time_summary = None
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]
        enable_amp = (
            torch.autocast(device_type="cuda", dtype=eval_precision)
            if torch.cuda.is_available()
            else nullcontext()
        )

        feature_dicts = [
            to_device(_copy_tree(data["input_feature_dict"]), self.device)
            for data in data_items
        ]
        token_counts = [
            int(feature_dict["residue_index"].shape[-1])
            for feature_dict in feature_dicts
        ]
        pad_token_trunk = len(set(token_counts)) > 1 and not exact_token_trunk
        max_tokens = max(token_counts)
        chunk_size = self.configs.infer_setting.chunk_size
        if (
            hasattr(self.configs.infer_setting, "dynamic_chunk_size")
            and self.configs.infer_setting.dynamic_chunk_size
        ):
            chunk_size = self.model._get_dynamic_chunk_size(max_tokens)
        inplace_safe = not (self.model.training or torch.is_grad_enabled())

        with enable_amp:
            prepared_features = []
            s_inputs_list = []
            for feature_dict in feature_dicts:
                feature_dict = self.model.relative_position_encoding.generate_relp(
                    feature_dict
                )
                feature_dict = update_input_feature_dict(feature_dict)
                s_inputs = self.model.input_embedder(
                    feature_dict, inplace_safe=False, chunk_size=chunk_size
                )
                prepared_features.append(feature_dict)
                s_inputs_list.append(s_inputs)

            if exact_token_trunk:
                # Padded-token pairformer batching is fast, but BF16/TF32
                # triangular reductions are not invariant to the padded physical
                # token length.  Running this trunk loop at each record's real
                # shape preserves the singleton pairformer boundary while the
                # denoising tail can still be batched below.
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                trunk_start = time.perf_counter()
                s_items = []
                z_items = []
                for feature_dict, s_inputs in zip(prepared_features, s_inputs_list):
                    s_i, z_i = self.model.get_pairformer_output_from_s_inputs(
                        input_feature_dict=_trunk_feature_dict(feature_dict),
                        s_inputs=s_inputs,
                        N_cycle=self.model.N_cycle,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                        pair_mask=None,
                    )
                    s_items.append(s_i)
                    z_items.append(z_i)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                trunk_sec_per_item = (
                    time.perf_counter() - trunk_start
                ) / len(data_items)
            else:
                if pad_token_trunk:
                    trunk_feature_dict = _stack_tree(
                        [
                            _pad_token_trunk_tree(
                                _trunk_feature_dict(feature_dict),
                                n_token=n_token_i,
                                max_tokens=max_tokens,
                            )
                            for feature_dict, n_token_i in zip(
                                prepared_features, token_counts
                            )
                        ]
                    )
                    s_inputs_batch = torch.stack(
                        [
                            _pad_tensor_token_axes(
                                s_inputs,
                                token_axes=(0,),
                                max_tokens=max_tokens,
                            )
                            for s_inputs in s_inputs_list
                        ],
                        dim=0,
                    )
                    pair_mask = _make_pair_mask(
                        token_counts=token_counts,
                        max_tokens=max_tokens,
                        device=s_inputs_batch.device,
                        dtype=s_inputs_batch.dtype,
                    )
                else:
                    trunk_feature_dict = _stack_tree(
                        [
                            _trunk_feature_dict(feature_dict)
                            for feature_dict in prepared_features
                        ]
                    )
                    s_inputs_batch = torch.stack(s_inputs_list, dim=0)
                    pair_mask = None
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                trunk_start = time.perf_counter()
                s_batch, z_batch = self.model.get_pairformer_output_from_s_inputs(
                    input_feature_dict=trunk_feature_dict,
                    s_inputs=s_inputs_batch,
                    N_cycle=self.model.N_cycle,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                    pair_mask=pair_mask,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                trunk_sec_per_item = (
                    time.perf_counter() - trunk_start
                ) / len(data_items)
                s_items = [
                    s_batch[idx, : token_counts[idx]]
                    if pad_token_trunk
                    else s_batch[idx]
                    for idx in range(len(data_items))
                ]
                z_items = [
                    z_batch[idx, : token_counts[idx], : token_counts[idx]]
                    if pad_token_trunk
                    else z_batch[idx]
                    for idx in range(len(data_items))
                ]

            use_batched_diffusion = _batched_token_diffusion_enabled(
                self.configs, len(data_items)
            )
            batched_coordinates = None
            diffusion_time_per_item = None
            trunk_source = "exact-trunk" if exact_token_trunk else "token-trunk"
            diffusion_batch_source = f"{trunk_source}-batch"
            if use_batched_diffusion:
                batched_coordinates, batch_diffusion_time = (
                    self.model.sample_diffusion_batch_token_transformer(
                        input_feature_dicts=prepared_features,
                        s_inputs_list=s_inputs_list,
                        s_list=s_items,
                        z_list=z_items,
                        chunk_size=chunk_size,
                        inplace_safe=inplace_safe,
                    )
                )
                batch_diffusion_time["diffusion_token_transformer_batched"] = True
                diffusion_batch_source = (
                    f"{trunk_source}+diffusion-token-atom-batch"
                    if batch_diffusion_time.get(
                        "diffusion_atom_transformer_batched", False
                    )
                    else f"{trunk_source}+diffusion-transformer-batch"
                )
                diffusion_time_per_item = _scale_time_dict(
                    batch_diffusion_time,
                    scale=1.0 / len(data_items),
                )

            predictions = []
            log_dicts = []
            for batch_idx, feature_dict in enumerate(prepared_features):
                s_i = s_items[batch_idx]
                z_i = z_items[batch_idx]
                pred_dict, log_dict, time_tracker = (
                    self.model.finish_inference_from_pairformer(
                        input_feature_dict=feature_dict,
                        s_inputs=s_inputs_list[batch_idx],
                        s=s_i,
                        z=z_i,
                        label_dict=None,
                        N_cycle=self.model.N_cycle,
                        mode="inference",
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                        pairformer_sec=trunk_sec_per_item,
                        precomputed_coordinate=(
                            batched_coordinates[batch_idx]
                            if batched_coordinates is not None
                            else None
                        ),
                        precomputed_diffusion_time=diffusion_time_per_item,
                    )
                )
                log_dict["time"] = time_tracker
                predictions.append(pred_dict)
                log_dicts.append(log_dict)

        self.last_log_dict = log_dicts[-1] if log_dicts else {}
        self.last_batch_time_summary = _summarize_model_time_dicts(
            [log_dict["time"] for log_dict in log_dicts if "time" in log_dict],
            item_count=len(data_items),
            source=diffusion_batch_source,
        )
        return predictions

    def print(self, msg: str) -> None:
        """
        Print message only on the master rank (rank 0).

        Args:
            msg (str): Message to print.
        """
        if DIST_WRAPPER.rank == 0:
            logger.info(msg)

    def update_model_configs(self, new_configs: Any) -> None:
        """
        Update the model's configuration.

        Args:
            new_configs (Any): New configuration object.
        """
        self.model.configs = new_configs


def _inference_batch_size(configs: Any) -> int:
    """Return the requested exact-shape inference batch size."""
    try:
        value = int(configs.get("inference_batch_size", 1))
    except Exception:
        value = 1
    return max(1, value)


def _inference_batch_mode(configs: Any) -> str:
    """Return how records are allowed to share one inference batch."""
    mode = str(configs.get("inference_batch_mode", "auto")).strip().lower()
    if mode not in {"auto", "exact", "token", "padded", "trunk_exact"}:
        logger.warning("Unknown inference_batch_mode=%r; using auto mode.", mode)
        return "auto"
    return mode


def _inference_token_bucket_size(configs: Any) -> int:
    """Return the approximate padded-token batching bucket width."""
    try:
        value = int(configs.get("inference_token_bucket_size", 0))
    except (TypeError, ValueError):
        logger.warning("Invalid inference_token_bucket_size; using input order.")
        return 0
    return max(0, value)


_ATOM_ONLY_FEATURE_KEYS = {
    "atom_to_token_idx",
    "atom_to_tokatom_idx",
    "bond_mask",
    "d_lm",
    "distogram_rep_atom_mask",
    "entity_mol_id",
    "is_dna",
    "is_ligand",
    "is_protein",
    "is_rna",
    "modified_res_mask",
    "mol_atom_index",
    "mol_id",
    "pad_info",
    "pae_rep_atom_mask",
    "plddt_m_rep_atom_mask",
    "ref_atom_name_chars",
    "ref_charge",
    "ref_element",
    "ref_mask",
    "ref_pos",
    "ref_space_uid",
    "v_lm",
}


def _copy_tree(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return {key: _copy_tree(sub_value) for key, sub_value in value.items()}
    if isinstance(value, list):
        return [_copy_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_tree(item) for item in value)
    return value


def _trunk_feature_dict(feature_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only features that can safely enter token-trunk batching."""
    return {
        key: value
        for key, value in feature_dict.items()
        if key not in _ATOM_ONLY_FEATURE_KEYS
    }


_TOKEN_AXIS_BY_KEY = {
    "asym_id": (0,),
    "deletion_mean": (0,),
    "entity_id": (0,),
    "esm_token_embedding": (0,),
    "frame_atom_index": (0,),
    "has_frame": (0,),
    "profile": (0,),
    "residue_index": (0,),
    "restype": (0,),
    "sym_id": (0,),
    "token_index": (0,),
    "has_deletion": (-1,),
    "deletion_value": (-1,),
    "msa": (-1,),
    "template_aatype": (-1,),
    "template_atom_mask": (-2,),
    "template_atom_positions": (-3,),
    "relp": (0, 1),
    "token_bonds": (0, 1),
    "template_backbone_frame_mask": (-2, -1),
    "template_distogram": (-3, -2),
    "template_pseudo_beta_mask": (-2, -1),
    "template_unit_vector": (-3, -2),
}


def _normalize_axes(axes: tuple[int, ...], ndim: int) -> tuple[int, ...]:
    return tuple(axis if axis >= 0 else ndim + axis for axis in axes)


def _token_axes_for_tensor(
    key_path: tuple[str, ...], tensor: torch.Tensor, n_token: int
) -> tuple[int, ...]:
    """Return the token-length axes for a feature tensor.

    The tempting generic rule, "pad every axis whose size equals N_token", is
    wrong for e.g. a 32-residue sequence where ``restype`` has shape
    ``[32, 32]``: the second axis is amino-acid class, not token position.  This
    small key map keeps padded-token batching explicit and reviewable.
    """
    key = key_path[-1] if key_path else ""
    axes = _TOKEN_AXIS_BY_KEY.get(key)
    if axes is not None:
        normalized = _normalize_axes(axes, tensor.ndim)
        if all(
            0 <= axis < tensor.ndim and tensor.shape[axis] == n_token
            for axis in normalized
        ):
            return normalized
        return ()
    if key_path and key_path[0] == "constraint_feature":
        return tuple(
            axis for axis, size in enumerate(tensor.shape) if size == n_token
        )
    return ()


def _padded_shape_signature(
    value: Any,
    n_token: int,
    key_path: tuple[str, ...] = (),
) -> Any:
    if isinstance(value, torch.Tensor):
        token_axes = set(_token_axes_for_tensor(key_path, value, n_token))
        shape = tuple(
            "N_TOKEN" if axis in token_axes else size
            for axis, size in enumerate(value.shape)
        )
        return ("tensor", shape, str(value.dtype))
    if isinstance(value, MappingABC):
        return (
            "dict",
            tuple(
                (
                    key,
                    _padded_shape_signature(
                        value[key], n_token, (*key_path, str(key))
                    ),
                )
                for key in sorted(value)
            ),
        )
    if isinstance(value, (list, tuple)):
        return (
            type(value).__name__,
            tuple(
                _padded_shape_signature(item, n_token, key_path)
                for item in value
            ),
        )
    return ("value", type(value).__name__, repr(value))


def _padded_token_trunk_signature(data: Mapping[str, Any]) -> Any:
    n_token = int(data["N_token"].item())
    return _padded_shape_signature(
        _trunk_feature_dict(data["input_feature_dict"]), n_token
    )


def _pad_tensor_token_axes(
    tensor: torch.Tensor,
    token_axes: tuple[int, ...],
    max_tokens: int,
) -> torch.Tensor:
    if not token_axes:
        return tensor
    out_shape = list(tensor.shape)
    changed = False
    for axis in token_axes:
        if out_shape[axis] > max_tokens:
            raise ValueError(
                f"cannot pad tensor shape {tuple(tensor.shape)} to {max_tokens}"
            )
        if out_shape[axis] != max_tokens:
            out_shape[axis] = max_tokens
            changed = True
    if not changed:
        return tensor
    padded = tensor.new_zeros(out_shape)
    slices = tuple(slice(0, size) for size in tensor.shape)
    padded[slices] = tensor
    return padded


def _pad_token_trunk_tree(
    value: Any,
    n_token: int,
    max_tokens: int,
    key_path: tuple[str, ...] = (),
) -> Any:
    if isinstance(value, torch.Tensor):
        return _pad_tensor_token_axes(
            value,
            token_axes=_token_axes_for_tensor(key_path, value, n_token),
            max_tokens=max_tokens,
        )
    if isinstance(value, MappingABC):
        return {
            key: _pad_token_trunk_tree(
                sub_value, n_token, max_tokens, (*key_path, str(key))
            )
            for key, sub_value in value.items()
        }
    if isinstance(value, list):
        return [
            _pad_token_trunk_tree(item, n_token, max_tokens, key_path)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _pad_token_trunk_tree(item, n_token, max_tokens, key_path)
            for item in value
        )
    return value


def _make_pair_mask(
    token_counts: list[int],
    max_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    token_mask = torch.zeros(
        (len(token_counts), max_tokens), device=device, dtype=dtype
    )
    for batch_idx, n_token in enumerate(token_counts):
        token_mask[batch_idx, :n_token] = 1
    return token_mask[..., :, None] * token_mask[..., None, :]


def _env_enabled_by_default(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "off", "no"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _guidance_enabled(configs: Any) -> bool:
    try:
        guidance = configs.sample_diffusion.to_dict().get("guidance")
    except AttributeError:
        guidance = getattr(configs.sample_diffusion, "guidance", None)
    if guidance is None:
        return False
    if isinstance(guidance, MappingABC):
        return bool(guidance.get("enable", False))
    return bool(getattr(guidance, "enable", False))


def _batched_token_diffusion_enabled(configs: Any, batch_size: int) -> bool:
    """Whether to share the diffusion token transformer across ragged records."""
    if batch_size <= 1:
        return False
    if not _env_enabled_by_default("PROTENIX_BATCH_DIFFUSION_TRANSFORMER"):
        return False
    n_sample = int(configs.sample_diffusion.N_sample)
    if n_sample < 1:
        return False
    # This ragged path is for low-sample design campaigns.  Very large
    # N_sample jobs already use the single-design fast path; mixing them with a
    # protein batch would multiply token/atom work and memory too aggressively.
    if n_sample > _env_int("PROTENIX_BATCH_DIFFUSION_MAX_SAMPLES", 5):
        return False
    if _guidance_enabled(configs):
        return False
    return True


def _scale_time_dict(time_dict: Mapping[str, Any], scale: float) -> dict[str, Any]:
    """Convert batch-total diffusion timings into per-record accounting.

    The new sampler runs one shared token-transformer launch per diffusion step.
    For batch summaries we want the sum of per-record timing dictionaries to be
    the true batch wall time, so numeric leaves are divided over records.
    """
    scaled = {}
    for key, value in time_dict.items():
        if isinstance(value, bool):
            scaled[key] = value
        elif isinstance(value, Number):
            scaled[key] = float(value) * scale
        else:
            scaled[key] = value
    return scaled


def _tensor_tree_signature(value: Any) -> Any:
    """Shape/dtype signature used to batch only exactly compatible inputs.

    Padding variable-length proteins changes the physical reduction lengths seen
    by attention and triangular kernels.  Even when masks prevent padded values
    from leaking into the valid region, different reduction schedules introduce
    tiny floating-point differences that are amplified by the 48-block trunk.
    Exact tensor-tree matching is deliberately conservative: it gives the GPU a
    leading batch dimension only when every model input already has the same
    physical shape, so batching is a throughput optimization rather than a
    numerical change.
    """
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), str(value.dtype))
    if isinstance(value, MappingABC):
        return (
            "dict",
            tuple((key, _tensor_tree_signature(value[key])) for key in sorted(value)),
        )
    if isinstance(value, (list, tuple)):
        return (
            type(value).__name__,
            tuple(_tensor_tree_signature(item) for item in value),
        )
    return ("value", type(value).__name__, repr(value))


def _input_feature_signature(data: Mapping[str, Any]) -> Any:
    return _tensor_tree_signature(data["input_feature_dict"])


def _token_trunk_signature(data: Mapping[str, Any]) -> Any:
    """Shape/dtype signature for the safe same-token trunk-batch boundary."""
    return _tensor_tree_signature(_trunk_feature_dict(data["input_feature_dict"]))


def _input_batch_signature(data: Mapping[str, Any], batch_mode: str) -> Any:
    if batch_mode in {"auto", "padded", "trunk_exact"}:
        return _padded_token_trunk_signature(data)
    if batch_mode == "token":
        return _token_trunk_signature(data)
    return _input_feature_signature(data)


def _token_bucket_id(n_token: int, bucket_size: int) -> int:
    return (n_token - 1) // bucket_size


def _queue_batch_signature(
    data: Mapping[str, Any], batch_mode: str, token_bucket_size: int
) -> Any:
    signature = _input_batch_signature(data, batch_mode)
    if batch_mode not in {"auto", "padded", "trunk_exact"} or token_bucket_size <= 0:
        return signature
    n_token = int(data["N_token"].item())
    return (signature, ("token_bucket", _token_bucket_id(n_token, token_bucket_size)))


def _effective_batch_mode(
    items: list[tuple[dict[str, Any], Any]], batch_mode: str
) -> str:
    """Choose the fastest safe execution boundary for an already grouped batch.

    ``auto`` is the practical campaign default.  It first keeps the older
    full-model exact-shape path when possible, then same-token trunk batching,
    then padded-token trunk batching.  Padded mode is the variable-sequence
    escape hatch: atom work stays ragged, while only token-trunk tensors are
    padded and protected by ``pair_mask``.
    """
    if len(items) <= 1 or batch_mode == "exact":
        return "exact"
    if batch_mode == "token":
        return "token"
    if batch_mode == "padded":
        return "padded"
    if batch_mode == "trunk_exact":
        return "trunk_exact"

    first_signature = _input_feature_signature(items[0][0])
    if all(
        _input_feature_signature(data) == first_signature for data, _ in items[1:]
    ):
        return "exact"
    first_token_signature = _token_trunk_signature(items[0][0])
    if all(
        _token_trunk_signature(data) == first_token_signature for data, _ in items[1:]
    ):
        return "token"
    return "padded"


def _stack_tree(items: list[Any]) -> Any:
    """Stack a homogeneous tensor tree along a new leading protein-batch axis."""
    first = items[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(items, dim=0)
    if isinstance(first, MappingABC):
        return {key: _stack_tree([item[key] for item in items]) for key in first}
    if isinstance(first, list):
        return [
            _stack_tree([item[index] for item in items])
            for index in range(len(first))
        ]
    if isinstance(first, tuple):
        return tuple(
            _stack_tree([item[index] for item in items])
            for index in range(len(first))
        )
    return first


def _stack_prediction_inputs(items: list[tuple[dict[str, Any], Any]]) -> dict[str, Any]:
    return {
        "input_feature_dict": _stack_tree(
            [data["input_feature_dict"] for data, _atom_array in items]
        )
    }


def _batched_prediction_n_sample(
    prediction: Mapping[str, Any],
    batch_size: int,
    default_n_sample: int,
) -> int:
    coordinate = prediction.get("coordinate")
    if (
        isinstance(coordinate, torch.Tensor)
        and coordinate.ndim >= 4
        and coordinate.shape[0] == batch_size
    ):
        return int(coordinate.shape[1])
    summary = prediction.get("summary_confidence")
    if isinstance(summary, list) and len(summary) % batch_size == 0:
        return max(1, len(summary) // batch_size)
    return max(1, default_n_sample)


def _slice_prediction_value(
    key: str,
    value: Any,
    batch_index: int,
    batch_size: int,
    n_sample: int,
) -> Any:
    """Take one protein from a batched prediction tree.

    Tensor predictions carry the protein batch as their leading dimension.
    Confidence summaries are flattened by ``compute_full_data_and_summary`` as
    ``batch * N_sample`` dictionaries, so they need a sample-sized list slice.
    """
    if key in {"summary_confidence", "full_data"} and isinstance(value, list):
        if len(value) % batch_size == 0:
            per_item = len(value) // batch_size
        else:
            per_item = n_sample
        start = batch_index * per_item
        return value[start : start + per_item]
    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and value.shape[0] == batch_size:
            return value[batch_index]
        return value
    if isinstance(value, MappingABC):
        return {
            sub_key: _slice_prediction_value(
                sub_key, sub_value, batch_index, batch_size, n_sample
            )
            for sub_key, sub_value in value.items()
        }
    return value


def _split_batched_prediction(
    prediction: Mapping[str, Any],
    batch_size: int,
    default_n_sample: int,
) -> list[dict[str, Any]]:
    n_sample = _batched_prediction_n_sample(prediction, batch_size, default_n_sample)
    return [
        {
            key: _slice_prediction_value(
                key, value, batch_index, batch_size, n_sample
            )
            for key, value in prediction.items()
        }
        for batch_index in range(batch_size)
    ]


def _write_inference_error(
    runner: InferenceRunner,
    sample_name: str,
    error_message: str,
) -> None:
    logger.error(error_message)
    with open(opjoin(runner.error_dir, f"{sample_name}.txt"), "a", encoding="utf-8") as f:
        f.write(error_message)


def _dump_prediction(
    runner: InferenceRunner,
    data: Mapping[str, Any],
    atom_array: Any,
    seed: int,
    prediction: Mapping[str, Any],
) -> None:
    runner.dumper.dump(
        dataset_name="",
        pdb_id=data["sample_name"],
        seed=seed,
        pred_dict=prediction,
        atom_array=atom_array,
        entity_poly_type={
            k: v for k, v in data["entity_poly_type"].items() if v != "non-polymer"
        },
    )


def _describe_batch(
    items: list[tuple[dict[str, Any], Any]], seed: int, batch_mode: str
) -> None:
    first_data = items[0][0]
    names = [data["sample_name"] for data, _atom_array in items]
    token_counts = [int(data["N_token"].item()) for data, _atom_array in items]
    atom_counts = [int(data["N_atom"].item()) for data, _atom_array in items]
    effective_mode = _effective_batch_mode(items, batch_mode)
    batch_kind_by_mode = {
        "exact": "same-shape",
        "token": "same-token-trunk",
        "padded": "padded-token-trunk",
        "trunk_exact": "singleton-token-trunk",
    }
    batch_kind = batch_kind_by_mode[effective_mode]
    if batch_mode == "auto":
        batch_kind = f"{batch_kind} (auto)"
    logger.info(
        "[Rank %s] Predicting %d %s input(s) [seed:%s]: "
        "N_token %s, token_pad_eff %.1f%%, N_atom %s-%s, N_msa %s, names=%s",
        DIST_WRAPPER.rank,
        len(items),
        batch_kind,
        seed,
        (
            str(token_counts[0])
            if min(token_counts) == max(token_counts)
            else f"{min(token_counts)}-{max(token_counts)}"
        ),
        100.0 * sum(token_counts) / (len(token_counts) * max(token_counts)),
        min(atom_counts),
        max(atom_counts),
        int(first_data["N_msa"].item()),
        ",".join(names[:4]) + ("..." if len(names) > 4 else ""),
    )


def _run_prediction_batch(
    runner: InferenceRunner,
    configs: Any,
    items: list[tuple[dict[str, Any], Any]],
    batch_mode: str,
) -> Mapping[str, Any] | list[Mapping[str, Any]]:
    first_data = items[0][0]
    n_token = max(int(data["N_token"].item()) for data, _atom_array in items)
    runner.update_model_configs(update_inference_configs(configs, n_token))
    effective_mode = _effective_batch_mode(items, batch_mode)
    if effective_mode in {"token", "padded", "trunk_exact"}:
        return runner.predict_token_batch(
            [data for data, _atom_array in items],
            exact_token_trunk=effective_mode == "trunk_exact",
        )
    batch_data = first_data if len(items) == 1 else _stack_prediction_inputs(items)
    return runner.predict(batch_data)


def _fallback_to_singletons(
    runner: InferenceRunner,
    configs: Any,
    items: list[tuple[dict[str, Any], Any]],
    seed: int,
    num_data: int,
    batch_mode: str,
    exc: Exception,
) -> None:
    if len(items) == 1:
        data = items[0][0]
        error_message = (
            f"[Rank {DIST_WRAPPER.rank}] {data['sample_name']} failed: {exc}\n"
            f"{traceback.format_exc()}"
        )
        _write_inference_error(runner, data["sample_name"], error_message)
        torch.cuda.empty_cache()
        return

    logger.warning(
        "Batched prediction of %d compatible inputs failed; falling back "
        "to singleton inference. Error: %s",
        len(items),
        exc,
    )
    torch.cuda.empty_cache()
    for item in items:
        _predict_and_dump_items(runner, configs, [item], seed, num_data, batch_mode)


def _prediction_items(
    prediction: Mapping[str, Any] | list[Mapping[str, Any]],
    configs: Any,
    batch_size: int,
) -> list[Mapping[str, Any]]:
    if isinstance(prediction, list):
        return prediction
    if batch_size == 1:
        return [prediction]
    return _split_batched_prediction(
        prediction,
        batch_size=batch_size,
        default_n_sample=int(configs.sample_diffusion.N_sample),
    )


_MODEL_TIME_KEYS = (
    "model_forward_with_summary",
    "model_forward",
    "pairformer",
    "diffusion",
    "diffusion_conditioning_sec",
    "diffusion_atom_encoder_sec",
    "diffusion_transformer_sec",
    "diffusion_atom_decoder_sec",
    "diffusion_input_scale_sec",
    "diffusion_output_rescale_sec",
    "contact_probs",
    "confidence",
    "confidence_head",
    "summary_confidence",
    "permutation",
)


def _timing_seconds(value: Any) -> float | None:
    """Convert scalar timing leaves to seconds, ignoring flags and strings.

    Protenix time dictionaries are intentionally lightweight: normal inference
    records Python floats, while multi-seed paths may merge scalars into small
    arrays.  This helper keeps logging tolerant without depending on numpy here.
    """
    if isinstance(value, bool):
        return None
    dtype = getattr(value, "dtype", None)
    if dtype is not None and str(dtype) in {"bool", "bool_", "torch.bool"}:
        return None
    if isinstance(value, Number):
        return float(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().sum().cpu().item())
    if hasattr(value, "sum"):
        try:
            return float(value.sum())
        except (TypeError, ValueError):
            return None
    return None


def _summarize_model_time_dicts(
    time_dicts: list[Mapping[str, Any]],
    item_count: int,
    source: str,
) -> dict[str, Any] | None:
    totals: dict[str, float] = {}
    for time_dict in time_dicts:
        for key, value in time_dict.items():
            seconds = _timing_seconds(value)
            if seconds is None:
                continue
            totals[key] = totals.get(key, 0.0) + seconds
    if not totals:
        return None
    per_input = {
        key: value / max(1, item_count)
        for key, value in totals.items()
    }
    return {
        "source": source,
        "items": item_count,
        "total": totals,
        "per_input": per_input,
    }


def _format_time_fields(values: Mapping[str, float], denominator: float | None) -> str:
    fields = []
    for key in _MODEL_TIME_KEYS:
        if key not in values:
            continue
        seconds = values[key]
        if denominator and denominator > 0:
            fields.append(f"{key}={seconds:.3f}s/{100.0 * seconds / denominator:.1f}%")
        else:
            fields.append(f"{key}={seconds:.3f}s")
    extra_keys = sorted(set(values) - set(_MODEL_TIME_KEYS))
    for key in extra_keys:
        fields.append(f"{key}={values[key]:.3f}s")
    return ", ".join(fields)


def _log_model_time_summary(
    runner: InferenceRunner,
    batch_size: int,
    predict_sec: float,
) -> None:
    summary = getattr(runner, "last_batch_time_summary", None)
    if summary is None:
        log_dict = getattr(runner, "last_log_dict", {})
        if isinstance(log_dict, MappingABC) and isinstance(
            log_dict.get("time"), MappingABC
        ):
            # In the exact same-shape path the model sees one leading batch
            # dimension, so the time dict is already the whole batch.  Dividing
            # by batch_size gives the per-record throughput view we care about.
            summary = _summarize_model_time_dicts(
                [log_dict["time"]],
                item_count=batch_size,
                source="exact-model-batch",
            )
    if summary is None:
        return

    total = summary["total"]
    per_input = summary["per_input"]
    logger.info(
        "[Rank %s] Batch model timing total (%s, %d input(s), predict %.2fs): %s",
        DIST_WRAPPER.rank,
        summary["source"],
        summary["items"],
        predict_sec,
        _format_time_fields(total, denominator=predict_sec),
    )
    if batch_size > 1:
        logger.info(
            "[Rank %s] Batch model timing per input (%s): %s",
            DIST_WRAPPER.rank,
            summary["source"],
            _format_time_fields(per_input, denominator=None),
        )


def _predict_and_dump_items(
    runner: InferenceRunner,
    configs: Any,
    items: list[tuple[dict[str, Any], Any]],
    seed: int,
    num_data: int,
    batch_mode: str,
) -> None:
    """Run one compatible batch, split outputs, and dump per input."""
    if not items:
        return

    batch_size = len(items)
    _describe_batch(items, seed, batch_mode)
    start = time.perf_counter()
    try:
        predict_start = time.perf_counter()
        prediction = _run_prediction_batch(runner, configs, items, batch_mode)
        if torch.cuda.is_available():
            # The dumper soon copies prediction tensors to CPU.  Synchronizing
            # here attributes completed GPU work to ``predict_sec`` instead of
            # hiding it inside file-writing time, without adding extra work to
            # normal dumped inference.
            torch.cuda.synchronize()
        predict_sec = time.perf_counter() - predict_start
        _log_model_time_summary(runner, batch_size, predict_sec)
    except Exception as exc:
        _fallback_to_singletons(
            runner, configs, items, seed, num_data, batch_mode, exc
        )
        return

    split_start = time.perf_counter()
    predictions = _prediction_items(prediction, configs, batch_size)
    split_sec = time.perf_counter() - split_start
    dump_sec = 0.0
    for prediction_i, (data, atom_array) in zip(predictions, items):
        dump_start = time.perf_counter()
        _dump_prediction(runner, data, atom_array, seed, prediction_i)
        dump_sec += time.perf_counter() - dump_start
        logger.info(
            "[Rank %s] %s [seed:%s] succeeded in batched forward. "
            "Results saved to %s",
            DIST_WRAPPER.rank,
            data["sample_name"],
            seed,
            configs.dump_dir,
        )
    logger.info(
        "[Rank %s] Finished %d/%d input(s) [seed:%s] in %.2fs "
        "(predict %.2fs, split %.2fs, dump %.2fs).",
        DIST_WRAPPER.rank,
        batch_size,
        num_data,
        seed,
        time.perf_counter() - start,
        predict_sec,
        split_sec,
        dump_sec,
    )
    torch.cuda.empty_cache()


def progress_callback(block_num: int, block_size: int, total_size: int) -> None:
    """Callback for tracking download progress."""
    downloaded = block_num * block_size
    percent = min(100, downloaded * 100 / total_size)
    bar_length = 30
    filled_length = int(bar_length * percent // 100)
    bar = "=" * filled_length + "-" * (bar_length - filled_length)

    status = f"\r[{bar}] {percent:.1f}%"
    print(status, end="", flush=True)

    if downloaded >= total_size:
        print()


def download_from_url(
    tos_url: str, checkpoint_path: str, check_weight: bool = True
) -> None:
    """Internal helper to download from URL and verify weight files."""
    urllib.request.urlretrieve(tos_url, checkpoint_path, reporthook=progress_callback)
    if check_weight:
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            del ckpt
        except Exception as e:
            if opexists(checkpoint_path):
                os.remove(checkpoint_path)
            raise RuntimeError(
                f"Download model checkpoint failed: {e}. Please download "
                f"manually with: wget {tos_url} -O {checkpoint_path}"
            ) from e


def download_inference_cache(configs: Any) -> None:
    """
    Download necessary data and model checkpoints for inference.

    Args:
        configs (Any): Configuration object containing paths and model names.
    """

    for cache_name in (
        "ccd_components_file",
        "ccd_components_rdkit_mol_file",
        "pdb_cluster_file",
        "obsolete_release_data_csv",
    ):
        cur_cache_fpath = configs["data"][cache_name]
        if not opexists(cur_cache_fpath):
            os.makedirs(os.path.dirname(cur_cache_fpath), exist_ok=True)
            tos_url = URL[cache_name]
            assert os.path.basename(tos_url) == os.path.basename(cur_cache_fpath), (
                f"{cache_name} file name is incorrect, `{tos_url}` and "
                f"`{cur_cache_fpath}`. Please check and try again."
            )
            logger.info(
                f"Downloading data cache from\n {tos_url}...\n to {cur_cache_fpath}"
            )
            download_from_url(tos_url, cur_cache_fpath, check_weight=False)

    if configs.use_template:
        for cache_name in (
            "obsolete_pdbs_path",
            "release_dates_path",
        ):
            cur_cache_fpath = configs["data"]["template"][cache_name]
            if not opexists(cur_cache_fpath):
                os.makedirs(os.path.dirname(cur_cache_fpath), exist_ok=True)
                tos_url = URL[cache_name]
                assert os.path.basename(tos_url) == os.path.basename(cur_cache_fpath), (
                    f"{cache_name} file name is incorrect, `{tos_url}` and "
                    f"`{cur_cache_fpath}`. Please check and try again."
                )
                logger.info(
                    f"Downloading data cache from\n {tos_url}...\n to {cur_cache_fpath}"
                )
                download_from_url(tos_url, cur_cache_fpath, check_weight=False)
            else:
                logger.info(f"{cache_name} already exists at {cur_cache_fpath}")

    checkpoint_path = f"{configs.load_checkpoint_dir}/{configs.model_name}.pt"
    checkpoint_dir = configs.load_checkpoint_dir

    if not opexists(checkpoint_path):
        os.makedirs(checkpoint_dir, exist_ok=True)
        tos_url = URL[configs.model_name]
        logger.info(
            f"Downloading model checkpoint from\n {tos_url}...\n to {checkpoint_path}"
        )
        download_from_url(tos_url, checkpoint_path)

    if "esm" in configs.model_name:  # currently esm only support 3b model
        esm_3b_ckpt_path = f"{checkpoint_dir}/esm2_t36_3B_UR50D.pt"
        if not opexists(esm_3b_ckpt_path):
            tos_url = URL["esm2_t36_3B_UR50D"]
            logger.info(
                f"Downloading model checkpoint from\n {tos_url}...\n to {esm_3b_ckpt_path}"
            )
            download_from_url(tos_url, esm_3b_ckpt_path)
        esm_3b_ckpt_path2 = f"{checkpoint_dir}/esm2_t36_3B_UR50D-contact-regression.pt"
        if not opexists(esm_3b_ckpt_path2):
            tos_url = URL["esm2_t36_3B_UR50D-contact-regression"]
            logger.info(
                f"Downloading model checkpoint from\n {tos_url}...\n to {esm_3b_ckpt_path2}"
            )
            download_from_url(tos_url, esm_3b_ckpt_path2)
    if "ism" in configs.model_name:
        esm_3b_ism_ckpt_path = f"{checkpoint_dir}/esm2_t36_3B_UR50D_ism.pt"

        if not opexists(esm_3b_ism_ckpt_path):
            tos_url = URL["esm2_t36_3B_UR50D_ism"]
            logger.info(
                f"Downloading model checkpoint from\n {tos_url}...\n to {esm_3b_ism_ckpt_path}"
            )
            download_from_url(tos_url, esm_3b_ism_ckpt_path)

        esm_3b_ism_ckpt_path2 = (
            f"{checkpoint_dir}/esm2_t36_3B_UR50D_ism-contact-regression.pt"
        )
        if not opexists(esm_3b_ism_ckpt_path2):
            tos_url = URL["esm2_t36_3B_UR50D_ism-contact-regression"]
            logger.info(
                f"Downloading model checkpoint from\n {tos_url}...\n to {esm_3b_ism_ckpt_path2}"
            )
            download_from_url(tos_url, esm_3b_ism_ckpt_path2)


def update_inference_configs(configs: Any, n_token: int) -> Any:
    """
    Adjust inference configurations based on the number of tokens to avoid OOM.

    Args:
        configs (Any): Original configurations.
        n_token (int): Number of tokens in the sample.

    Returns:
        Any: Updated configurations.
    """
    # Adjust configurations based on sequence length to manage memory usage
    if n_token > 2560 and configs.model_name in ["protenix-v2"]:
        raise AssertionError(
            "protenix-v2 model does not support n_token > 2560. It might cause OOM."
        )

    if n_token > 3840:
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = False
    elif n_token > 2560:
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = True
    else:
        if configs.model_name in ["protenix-v2"]:
            configs.skip_amp.confidence_head = False
        else:
            configs.skip_amp.confidence_head = True
        configs.skip_amp.sample_diffusion = True

    return configs


def infer_predict(runner: InferenceRunner, configs: Any) -> None:
    """
    Run the full inference process for the given runner and configurations.
    Processes all samples in the dataloader for each specified seed.

    Args:
        runner (InferenceRunner): The initialized runner instance.
        configs (Any): Inference configurations.
    """
    # Data loading
    logger.info(f"Loading data from {configs.input_json_path}")
    with open(configs.input_json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    if not isinstance(json_data, list) or len(json_data) == 0:
        raise ValueError(
            f"Input JSON must be a non-empty top-level list, got {type(json_data).__name__} "
            f"from {configs.input_json_path}"
        )

    seed_in_json = json_data[0].get("modelSeeds")
    if seed_in_json and configs.use_seeds_in_json:
        seeds = [int(i) for i in seed_in_json]
        logger.info(f"Using seeds from JSON: {seeds}")
    else:
        seeds = configs.seeds

    try:
        dataloader = get_inference_dataloader(configs=configs)
    except Exception as e:
        error_message = (
            f"Dataloader initialization failed: {e}\n{traceback.format_exc()}"
        )
        logger.error(error_message)
        with open(opjoin(runner.error_dir, "error.txt"), "a", encoding="utf-8") as f:
            f.write(error_message)
        return

    num_data = len(dataloader.dataset)
    t0_start = time.time()
    for seed in seeds:
        seed_everything(seed=seed, deterministic=configs.deterministic)
        t1_start = time.time()
        batch_size = _inference_batch_size(configs)
        batch_mode = _inference_batch_mode(configs)
        token_bucket_size = _inference_token_bucket_size(configs)
        pending: dict[Any, list[tuple[dict[str, Any], Any]]] = {}

        def flush_signature(signature: Any) -> None:
            items = pending.pop(signature, [])
            _predict_and_dump_items(
                runner=runner,
                configs=configs,
                items=items,
                seed=seed,
                num_data=num_data,
                batch_mode=batch_mode,
            )

        for batch in dataloader:
            for data, atom_array, data_error_message in batch:
                sample_name = data.get("sample_name", "unknown")
                try:
                    if len(data_error_message) > 0:
                        _write_inference_error(
                            runner,
                            sample_name,
                            f"Data error for {sample_name}: {data_error_message}",
                        )
                        continue

                    logger.info(
                        f"[Rank {DIST_WRAPPER.rank} ({data['sample_index'] + 1}/{num_data})] "
                        f"{sample_name} [seed:{seed}]: "
                        f"N_asym {data['N_asym'].item()}, N_token {data['N_token'].item()}, "
                        f"N_atom {data['N_atom'].item()}, N_msa {data['N_msa'].item()}"
                    )
                    signature = _queue_batch_signature(
                        data, batch_mode, token_bucket_size
                    )
                    pending.setdefault(signature, []).append((data, atom_array))
                    if len(pending[signature]) >= batch_size:
                        flush_signature(signature)
                except Exception as e:
                    error_message = (
                        f"[Rank {DIST_WRAPPER.rank}] {sample_name} failed: {e}\n"
                        f"{traceback.format_exc()}"
                    )
                    _write_inference_error(runner, sample_name, error_message)
                    torch.cuda.empty_cache()

        for signature in list(pending):
            flush_signature(signature)
        t1_end = time.time()
        logger.info(
            f"[Rank {DIST_WRAPPER.rank}] Seed {seed} completed in {t1_end - t1_start:.2f}s."
        )
    # Remove the error directory if it's empty
    if opexists(runner.error_dir):
        try:
            if not os.listdir(runner.error_dir):
                os.rmdir(runner.error_dir)
        except Exception:
            pass

    t0_end = time.time()
    logger.info(
        f"[Rank {DIST_WRAPPER.rank}] Job completed in {t0_end - t0_start:.2f}s."
    )


def main(configs: Any) -> None:
    """
    Inference entry point.

    Args:
        configs (Any): Inference configurations.
    """
    runner = InferenceRunner(configs)
    infer_predict(runner, configs)


def update_gpu_compatible_configs(configs: Any) -> Any:
    """
    Update configurations to ensure compatibility with specific GPU architectures (e.g., V100).

    Args:
        configs (Any): Original configurations.

    Returns:
        Any: Updated configurations.
    """

    def is_gpu_capability_between_7_and_8() -> bool:
        # Check if 7.0 <= device_capability < 8.0
        if not torch.cuda.is_available():
            return False
        capability = torch.cuda.get_device_capability()
        major, minor = capability
        cc = major + minor / 10.0
        return 7.0 <= cc < 8.0

    if is_gpu_capability_between_7_and_8():
        # V100 and similar architectures don't support some kernels or BF16 effectively
        configs.dtype = "fp32"
        configs.triangle_attention = "torch"
        configs.triangle_multiplicative = "torch"
        logger.info(
            "Enforcing FP32 and torch kernels for compatibility with detected "
            "GPU (Compute Capability 7.x)."
        )
    return configs


def run() -> None:
    """
    Initialize and execute the inference pipeline.
    """
    log_format = (
        "%(asctime)s,%(msecs)-3d %(levelname)-8s "
        "[%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    )
    logging.basicConfig(
        format=log_format,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )

    arg_str = parse_sys_args()
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    # 1. First pass to get model_name
    configs = parse_configs(
        configs=configs,
        arg_str=arg_str,
        fill_required_with_null=True,
    )
    model_name = configs.model_name

    # 2. Get model specifics and merge into base defaults
    base_configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    model_specfics_configs = model_configs[model_name]

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d

    deep_update(base_configs, model_specfics_configs)

    # 3. Second pass to apply sys_args with higher priority
    configs = parse_configs(
        configs=base_configs,
        arg_str=arg_str,
        fill_required_with_null=True,
    )
    logger.info(
        f"Using params for model {model_name}: "
        f"cycle={configs.model.N_cycle}, step={configs.sample_diffusion.N_step}"
    )
    model_name_parts = model_name.split("_", 3)
    if len(model_name_parts) == 4:
        _, model_size, model_feature, model_version = model_name_parts
    elif model_name == "protenix-v2":
        # The model naming convention has been simplified for newer versions.
        # Hardcoding these values here to maintain backward compatibility.
        model_size = "464M"
        model_feature = "default"
        model_version = "v2"
    else:
        model_size = "unknown"
        model_feature = "unknown"
        model_version = "unknown"
        logger.warning(
            "Unexpected model_name format '%s'; expected protenix_<size>_<feature>_<version>.",
            model_name,
        )
    logger.info(
        f"Inference by Protenix: model_size: {model_size}, "
        f"with_feature: {model_feature.replace('-', ', ')}, "
        f"model_version: {model_version}, dtype: {configs.dtype}"
    )
    configs = update_gpu_compatible_configs(configs)
    logger.info(
        f"Triangle kernels: multiplicative={configs.triangle_multiplicative}, "
        f"attention={configs.triangle_attention}"
    )
    logger.info(
        f"Optimization: shared_vars_cache={configs.enable_diffusion_shared_vars_cache}, "
        f"efficient_fusion={configs.enable_efficient_fusion}, tf32={configs.enable_tf32}"
    )
    download_inference_cache(configs)
    main(configs)


if __name__ == "__main__":
    run()
