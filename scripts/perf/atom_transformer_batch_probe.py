#!/usr/bin/env python3
"""Screen padded batching for the atom-local transformer.

After batching the token-level diffusion transformer, mixed-token inference is
mostly limited by the atom encoder/decoder.  Full atom padding is only safe if
fake atoms are invisible as local-attention keys and excluded from atom-to-token
means.  This probe exercises that new mask boundary on the core
``AtomTransformer`` before integrating it into the full sampler.
"""

from __future__ import annotations

import _repo_bootstrap  # noqa: F401

import argparse
import statistics
from collections.abc import Callable

import torch

from protenix.model.modules.transformer import AtomTransformer


def _parse_lengths(value: str) -> list[int]:
    lengths = [int(part) for part in value.split(",") if part]
    if not lengths:
        raise argparse.ArgumentTypeError("at least one atom length is required")
    return lengths


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _event_time_ms(fn: Callable[[], object], warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times


def _random_records(
    lengths: list[int],
    c_atom: int,
    c_atompair: int,
    n_queries: int,
    n_keys: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    records = []
    for n_atom in lengths:
        n_blocks = _ceil_div(n_atom, n_queries)
        records.append(
            (
                torch.randn(n_atom, c_atom, device=device),
                torch.randn(n_atom, c_atom, device=device),
                torch.randn(
                    n_blocks,
                    n_queries,
                    n_keys,
                    c_atompair,
                    device=device,
                ),
            )
        )
    return records


def _padded_inputs(
    records: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    c_atom: int,
    c_atompair: int,
    n_queries: int,
    n_keys: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(records)
    max_atoms = max(record[0].shape[0] for record in records)
    max_blocks = _ceil_div(max_atoms, n_queries)
    device = records[0][0].device
    q = torch.zeros(batch, max_atoms, c_atom, device=device)
    c = torch.zeros(batch, max_atoms, c_atom, device=device)
    p = torch.zeros(batch, max_blocks, n_queries, n_keys, c_atompair, device=device)
    atom_mask = torch.zeros(batch, max_atoms, device=device, dtype=torch.bool)
    for idx, (q_i, c_i, p_i) in enumerate(records):
        n_atom = q_i.shape[0]
        n_blocks = p_i.shape[0]
        q[idx, :n_atom] = q_i
        c[idx, :n_atom] = c_i
        p[idx, :n_blocks] = p_i
        atom_mask[idx, :n_atom] = True
    return q, c, p, atom_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lengths",
        type=_parse_lengths,
        default="201,201,261,261,321,321,381,381",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--c-atom", type=int, default=128)
    parser.add_argument("--c-atompair", type=int, default=16)
    parser.add_argument("--n-queries", type=int, default=32)
    parser.add_argument("--n-keys", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This probe requires CUDA.")

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    model = AtomTransformer(
        n_blocks=args.blocks,
        n_heads=args.heads,
        c_atom=args.c_atom,
        c_atompair=args.c_atompair,
        n_queries=args.n_queries,
        n_keys=args.n_keys,
    ).to(device)
    model.eval()

    records = _random_records(
        args.lengths,
        args.c_atom,
        args.c_atompair,
        args.n_queries,
        args.n_keys,
        device,
    )
    q_batch, c_batch, p_batch, atom_mask = _padded_inputs(
        records,
        args.c_atom,
        args.c_atompair,
        args.n_queries,
        args.n_keys,
    )

    @torch.no_grad()
    def run_sequential() -> list[torch.Tensor]:
        outputs = []
        for q_i, c_i, p_i in records:
            outputs.append(
                model(
                    q_i,
                    c_i,
                    p_i,
                    inplace_safe=True,
                )
            )
        return outputs

    @torch.no_grad()
    def run_padded() -> torch.Tensor:
        return model(
            q_batch,
            c_batch,
            p_batch,
            inplace_safe=True,
            atom_mask=atom_mask,
        )

    seq_times = _event_time_ms(lambda: run_sequential(), args.warmup, args.iters)
    pad_times = _event_time_ms(lambda: run_padded(), args.warmup, args.iters)

    seq_out = run_sequential()
    pad_out = run_padded()
    max_abs = 0.0
    for idx, out_i in enumerate(seq_out):
        n_atom = out_i.shape[-2]
        diff = (out_i - pad_out[idx, :n_atom]).float().abs().max().item()
        max_abs = max(max_abs, diff)

    seq_med = statistics.median(seq_times)
    pad_med = statistics.median(pad_times)
    print(f"lengths={args.lengths}")
    print(f"sequential_ms_median={seq_med:.3f}")
    print(f"padded_batch_ms_median={pad_med:.3f}")
    print(f"speedup={seq_med / pad_med:.3f}x")
    print(f"valid_prefix_max_abs_diff={max_abs:.6g}")
    print(f"sequential_ms={','.join(f'{t:.3f}' for t in seq_times)}")
    print(f"padded_batch_ms={','.join(f'{t:.3f}' for t in pad_times)}")


if __name__ == "__main__":
    main()
