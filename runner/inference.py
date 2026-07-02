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
from protenix.model.protenix import Protenix
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
    mode = str(configs.get("inference_batch_mode", "exact")).lower()
    if mode not in {"exact", "padded"}:
        logger.warning("Unknown inference_batch_mode=%r; using exact batching", mode)
        return "exact"
    return mode


def _inference_batch_max_padding_fraction(configs: Any) -> float:
    try:
        value = float(configs.get("inference_batch_max_padding_fraction", 0.25))
    except Exception:
        value = 0.25
    return min(1.0, max(0.0, value))


_TOKEN_AXIS0_FEATURES = {
    "asym_id",
    "deletion_mean",
    "entity_id",
    "frame_atom_index",
    "has_frame",
    "profile",
    "residue_index",
    "restype",
    "sym_id",
    "token_index",
}
_ATOM_AXIS0_FEATURES = {
    "atom_to_tokatom_idx",
    "atom_to_token_idx",
    "distogram_rep_atom_mask",
    "entity_mol_id",
    "is_dna",
    "is_ligand",
    "is_protein",
    "is_rna",
    "modified_res_mask",
    "mol_atom_index",
    "mol_id",
    "pae_rep_atom_mask",
    "plddt_m_rep_atom_mask",
    "ref_atom_name_chars",
    "ref_charge",
    "ref_element",
    "ref_mask",
    "ref_pos",
    "ref_space_uid",
}
_TOKEN_PAIR_FEATURES = {
    "token_bonds",
    "constraint_feature.contact",
    "constraint_feature.contact_atom",
    "constraint_feature.pocket",
    "constraint_feature.substructure",
}


def _feature_path(path: tuple[Any, ...]) -> str:
    if len(path) >= 2 and path[0] == "input_feature_dict":
        return ".".join(str(part) for part in path[1:])
    return ""


def _pad_axes_for_path(path: tuple[Any, ...], tensor: torch.Tensor) -> tuple[int, ...]:
    """Return variable axes that may be padded for an inference feature.

    The key point is to be explicit: token/atom axes are physical sequence
    dimensions, while channel/class/template axes must stay fixed. Unknown
    tensors are still batchable when their shapes already match exactly.
    """
    feature = _feature_path(path)
    if tensor.ndim == 0:
        return ()
    if feature in _TOKEN_AXIS0_FEATURES:
        return (0,)
    if feature in _ATOM_AXIS0_FEATURES:
        return (0,)
    if feature in _TOKEN_PAIR_FEATURES:
        return (0, 1)
    if feature == "bond_mask":
        return (0, 1)
    if feature in {"deletion_value", "has_deletion", "msa"}:
        # Token columns may be padded. MSA row counts are intentionally kept
        # fixed for this first public path because row-padding semantics are
        # less obvious than token/atom masking.
        return (1,)
    if feature == "template_aatype":
        return (1,)
    if feature in {"template_atom_mask", "template_atom_positions"}:
        return (1,)
    if feature in {
        "template_backbone_frame_mask",
        "template_distogram",
        "template_pseudo_beta_mask",
        "template_unit_vector",
    }:
        return (1, 2)
    return ()


def _tensor_tree_signature(value: Any, path: tuple[Any, ...] = ()) -> Any:
    """Shape/dtype signature used to batch only exactly compatible inputs.

    Padding variable-length proteins is not generally equivalent for this model:
    several triangular operations can let padded tokens affect the valid region.
    Exact tensor-tree matching is deliberately conservative. It gives the GPU a
    leading batch dimension only when every model input already has the same
    physical shape.
    """
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), str(value.dtype))
    if isinstance(value, MappingABC):
        return (
            "dict",
            tuple(
                (key, _tensor_tree_signature(value[key], (*path, key)))
                for key in sorted(value)
            ),
        )
    if isinstance(value, (list, tuple)):
        return (
            type(value).__name__,
            tuple(
                _tensor_tree_signature(item, (*path, index))
                for index, item in enumerate(value)
            ),
        )
    return ("value", type(value).__name__, repr(value))


def _padded_tensor_tree_signature(value: Any, path: tuple[Any, ...] = ()) -> Any:
    if isinstance(value, torch.Tensor):
        pad_axes = set(_pad_axes_for_path(path, value))
        shape = tuple(
            None if axis in pad_axes else size
            for axis, size in enumerate(value.shape)
        )
        return ("tensor", value.ndim, shape, str(value.dtype))
    if isinstance(value, MappingABC):
        return (
            "dict",
            tuple(
                (key, _padded_tensor_tree_signature(value[key], (*path, key)))
                for key in sorted(value)
            ),
        )
    if isinstance(value, (list, tuple)):
        return (
            type(value).__name__,
            tuple(
                _padded_tensor_tree_signature(item, (*path, index))
                for index, item in enumerate(value)
            ),
        )
    return ("value", type(value).__name__, repr(value))


def _input_feature_signature(data: Mapping[str, Any]) -> Any:
    return _tensor_tree_signature(data["input_feature_dict"])


def _padded_input_feature_signature(data: Mapping[str, Any]) -> Any:
    return _padded_tensor_tree_signature(
        {"input_feature_dict": data["input_feature_dict"]}
    )


def _batch_signature(data: Mapping[str, Any], mode: str) -> Any:
    if mode == "padded":
        return _padded_input_feature_signature(data)
    return _input_feature_signature(data)


def _feature_token_count(data: Mapping[str, Any]) -> int:
    return int(data["N_token"].item())


def _feature_atom_count(data: Mapping[str, Any]) -> int:
    return int(data["N_atom"].item())


def _batch_padding_fraction(items: list[tuple[dict[str, Any], Any]]) -> float:
    if len(items) <= 1:
        return 0.0
    token_counts = [_feature_token_count(data) for data, _atom_array in items]
    max_tokens = max(token_counts)
    if max_tokens <= 0:
        return 0.0
    useful_pair_cells = sum(tokens * tokens for tokens in token_counts)
    padded_pair_cells = len(items) * max_tokens * max_tokens
    return 1.0 - (useful_pair_cells / padded_pair_cells)


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


def _pad_tensor_to_shape(
    tensor: torch.Tensor,
    target_shape: tuple[int, ...],
    path: tuple[Any, ...],
) -> torch.Tensor:
    if tuple(tensor.shape) == target_shape:
        return tensor
    pad_axes = set(_pad_axes_for_path(path, tensor))
    for axis, (size, target) in enumerate(zip(tensor.shape, target_shape)):
        if size != target and axis not in pad_axes:
            raise ValueError(
                f"Cannot pad feature {'.'.join(map(str, path))}: "
                f"axis {axis} is fixed ({size} != {target})"
            )
    out = tensor.new_zeros(target_shape)
    out[tuple(slice(0, size) for size in tensor.shape)] = tensor
    return out


def _pad_stack_tree(items: list[Any], path: tuple[Any, ...] = ()) -> Any:
    """Stack a feature tree, padding only the axes declared by feature key."""
    first = items[0]
    if isinstance(first, torch.Tensor):
        target_shape = tuple(
            max(item.shape[axis] for item in items) for axis in range(first.ndim)
        )
        return torch.stack(
            [_pad_tensor_to_shape(item, target_shape, path) for item in items],
            dim=0,
        )
    if isinstance(first, MappingABC):
        return {
            key: _pad_stack_tree([item[key] for item in items], (*path, key))
            for key in first
        }
    if isinstance(first, list):
        return [
            _pad_stack_tree([item[index] for item in items], (*path, index))
            for index in range(len(first))
        ]
    if isinstance(first, tuple):
        return tuple(
            _pad_stack_tree([item[index] for item in items], (*path, index))
            for index in range(len(first))
        )
    return first


def _stack_prediction_inputs(
    items: list[tuple[dict[str, Any], Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    feature_items = [data["input_feature_dict"] for data, _atom_array in items]
    if mode == "padded":
        feature_dict = _pad_stack_tree(feature_items, ("input_feature_dict",))
    else:
        feature_dict = _stack_tree(feature_items)
    return {
        "input_feature_dict": feature_dict,
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


def _slice_tensor_axis(value: torch.Tensor, axis: int, length: int) -> torch.Tensor:
    if value.ndim == 0:
        return value
    axis = axis if axis >= 0 else value.ndim + axis
    if axis < 0 or axis >= value.ndim or value.shape[axis] <= length:
        return value
    slices = [slice(None)] * value.ndim
    slices[axis] = slice(0, length)
    return value[tuple(slices)]


def _trim_prediction_tensor(
    key: str,
    value: torch.Tensor,
    *,
    n_token: int,
    n_atom: int,
) -> torch.Tensor:
    """Remove padded token/atom tails from common prediction tensors."""
    if key in {
        "coordinate",
        "atom_coordinate",
        "atom_plddt",
        "atom_to_token_idx",
        "atom_is_polymer",
        "plddt",
        "resolved",
    }:
        # Atom-shaped tensors are either [N_sample, N_atom, ...] or [N_atom].
        atom_axis = -2 if value.ndim >= 2 and value.shape[-1] != n_atom else -1
        return _slice_tensor_axis(value, atom_axis, n_atom)
    if key in {
        "pae",
        "pde",
        "contact_probs",
        "token_pair_pae",
        "token_pair_pde",
    }:
        # Token-pair tensors are [N, N], [N_sample, N, N], or logits with a
        # final bin channel [N_sample, N, N, C].
        if value.ndim >= 4:
            value = _slice_tensor_axis(value, -3, n_token)
            return _slice_tensor_axis(value, -2, n_token)
        if value.ndim >= 2:
            value = _slice_tensor_axis(value, -2, n_token)
            return _slice_tensor_axis(value, -1, n_token)
    if key in {"token_has_frame", "token_asym_id"}:
        return _slice_tensor_axis(value, -1, n_token)
    return value


def _trim_prediction_value(
    key: str,
    value: Any,
    *,
    n_token: int,
    n_atom: int,
) -> Any:
    if isinstance(value, torch.Tensor):
        return _trim_prediction_tensor(key, value, n_token=n_token, n_atom=n_atom)
    if isinstance(value, MappingABC):
        return {
            sub_key: _trim_prediction_value(
                sub_key, sub_value, n_token=n_token, n_atom=n_atom
            )
            for sub_key, sub_value in value.items()
        }
    if isinstance(value, list):
        return [
            _trim_prediction_value(key, item, n_token=n_token, n_atom=n_atom)
            for item in value
        ]
    return value


def _trim_prediction_for_data(
    prediction: Mapping[str, Any],
    data: Mapping[str, Any],
) -> dict[str, Any]:
    n_token = _feature_token_count(data)
    n_atom = _feature_atom_count(data)
    return {
        key: _trim_prediction_value(key, value, n_token=n_token, n_atom=n_atom)
        for key, value in prediction.items()
    }


def _write_inference_error(
    runner: InferenceRunner,
    sample_name: str,
    error_message: str,
) -> None:
    logger.error(error_message)
    with open(
        opjoin(runner.error_dir, f"{sample_name}.txt"), "a", encoding="utf-8"
    ) as f:
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
    items: list[tuple[dict[str, Any], Any]],
    seed: int,
    *,
    mode: str,
) -> None:
    first_data = items[0][0]
    names = [data["sample_name"] for data, _atom_array in items]
    token_counts = [_feature_token_count(data) for data, _atom_array in items]
    atom_counts = [_feature_atom_count(data) for data, _atom_array in items]
    logger.info(
        "[Rank %s] Predicting %d %s input(s) [seed:%s]: "
        "N_token %s, N_atom %s, N_msa %s, padding_fraction %.3f, names=%s",
        DIST_WRAPPER.rank,
        len(items),
        "padded" if mode == "padded" and len(items) > 1 else "same-shape",
        seed,
        (
            f"{min(token_counts)}-{max(token_counts)}"
            if min(token_counts) != max(token_counts)
            else str(token_counts[0])
        ),
        (
            f"{min(atom_counts)}-{max(atom_counts)}"
            if min(atom_counts) != max(atom_counts)
            else str(atom_counts[0])
        ),
        int(first_data["N_msa"].item()),
        _batch_padding_fraction(items) if mode == "padded" else 0.0,
        ",".join(names[:4]) + ("..." if len(names) > 4 else ""),
    )


def _run_prediction_batch(
    runner: InferenceRunner,
    configs: Any,
    items: list[tuple[dict[str, Any], Any]],
) -> Mapping[str, Any]:
    mode = _inference_batch_mode(configs)
    n_token = max(_feature_token_count(data) for data, _atom_array in items)
    runner.update_model_configs(update_inference_configs(configs, n_token))
    first_data = items[0][0]
    batch_data = (
        first_data
        if len(items) == 1
        else _stack_prediction_inputs(items, mode=mode)
    )
    return runner.predict(batch_data)


def _fallback_to_singletons(
    runner: InferenceRunner,
    configs: Any,
    items: list[tuple[dict[str, Any], Any]],
    seed: int,
    num_data: int,
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
        "Batched prediction of %d same-shape inputs failed; falling back "
        "to singleton inference. Error: %s",
        len(items),
        exc,
    )
    torch.cuda.empty_cache()
    for item in items:
        _predict_and_dump_items(runner, configs, [item], seed, num_data)


def _prediction_items(
    prediction: Mapping[str, Any],
    configs: Any,
    items: list[tuple[dict[str, Any], Any]],
) -> list[Mapping[str, Any]]:
    batch_size = len(items)
    if batch_size == 1:
        return [prediction]
    predictions = _split_batched_prediction(
        prediction,
        batch_size=batch_size,
        default_n_sample=int(configs.sample_diffusion.N_sample),
    )
    if _inference_batch_mode(configs) == "padded":
        return [
            _trim_prediction_for_data(prediction_i, data)
            for prediction_i, (data, _atom_array) in zip(predictions, items)
        ]
    return predictions


def _predict_and_dump_items(
    runner: InferenceRunner,
    configs: Any,
    items: list[tuple[dict[str, Any], Any]],
    seed: int,
    num_data: int,
) -> None:
    """Run one exact-shape batch, split outputs, and dump per input."""
    if not items:
        return

    batch_size = len(items)
    mode = _inference_batch_mode(configs)
    _describe_batch(items, seed, mode=mode)
    start = time.time()
    try:
        prediction = _run_prediction_batch(runner, configs, items)
    except Exception as exc:
        _fallback_to_singletons(runner, configs, items, seed, num_data, exc)
        return

    predictions = _prediction_items(prediction, configs, items)
    for prediction_i, (data, atom_array) in zip(predictions, items):
        _dump_prediction(runner, data, atom_array, seed, prediction_i)
        logger.info(
            "[Rank %s] %s [seed:%s] succeeded in batched forward. "
            "Results saved to %s",
            DIST_WRAPPER.rank,
            data["sample_name"],
            seed,
            configs.dump_dir,
        )
    logger.info(
        "[Rank %s] Finished %d/%d input(s) [seed:%s] in %.2fs.",
        DIST_WRAPPER.rank,
        batch_size,
        num_data,
        seed,
        time.time() - start,
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
        max_padding_fraction = _inference_batch_max_padding_fraction(configs)
        pending: dict[Any, list[tuple[dict[str, Any], Any]]] = {}

        def flush_signature(signature: Any) -> None:
            items = pending.pop(signature, [])
            _predict_and_dump_items(
                runner=runner,
                configs=configs,
                items=items,
                seed=seed,
                num_data=num_data,
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
                    signature = _batch_signature(data, batch_mode)
                    item = (data, atom_array)
                    bucket = pending.setdefault(signature, [])
                    if (
                        batch_mode == "padded"
                        and bucket
                        and _batch_padding_fraction([*bucket, item])
                        > max_padding_fraction
                    ):
                        flush_signature(signature)
                        bucket = pending.setdefault(signature, [])
                    bucket.append(item)
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
