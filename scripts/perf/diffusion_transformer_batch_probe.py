#!/usr/bin/env python3
"""Screen padded batching for the diffusion token transformer.

The mixed-token inference path already batches the pairformer trunk, then runs
the diffusion/confidence tail per record because atom tensors are ragged.  The
N_step=200 profile shows the largest remaining hotspot is not atom attention but
the token-level diffusion transformer.  This probe isolates that boundary:

* sequential: run each variable-length token transformer at its exact length;
* padded: pad tokens to the largest length, pass a real token key mask, and run
  one batched transformer call.

This is not an end-to-end gate.  It is a cheap way to decide whether the more
invasive sampler integration is worth doing.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable

import torch

from protenix.model.modules.transformer import DiffusionTransformer


def _parse_lengths(value: str) -> list[int]:
    lengths = [int(part) for part in value.split(",") if part]
    if not lengths:
        raise argparse.ArgumentTypeError("at least one token length is required")
    return lengths


def _event_time_ms(fn: Callable[[], None], warmup: int, iters: int) -> list[float]:
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


def _random_inputs(
    lengths: list[int],
    c_a: int,
    c_s: int,
    c_z: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    records = []
    for n_token in lengths:
        records.append(
            (
                torch.randn(n_token, c_a, device=device),
                torch.randn(n_token, c_s, device=device),
                torch.randn(n_token, n_token, c_z, device=device),
            )
        )
    return records


def _padded_inputs(
    records: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    c_a: int,
    c_s: int,
    c_z: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(records)
    max_tokens = max(record[0].shape[0] for record in records)
    device = records[0][0].device
    a = torch.zeros(batch, max_tokens, c_a, device=device)
    s = torch.zeros(batch, max_tokens, c_s, device=device)
    z = torch.zeros(batch, max_tokens, max_tokens, c_z, device=device)
    token_mask = torch.zeros(batch, max_tokens, device=device, dtype=torch.bool)
    for idx, (a_i, s_i, z_i) in enumerate(records):
        n_token = a_i.shape[0]
        a[idx, :n_token] = a_i
        s[idx, :n_token] = s_i
        z[idx, :n_token, :n_token] = z_i
        token_mask[idx, :n_token] = True
    return a, s, z, token_mask


def _channel_first_z(z: torch.Tensor) -> torch.Tensor:
    return z.movedim(-1, -3).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lengths",
        type=_parse_lengths,
        default="40,52,64,76,88,100,112,124",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--blocks", type=int, default=24)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--c-a", type=int, default=768)
    parser.add_argument("--c-s", type=int, default=384)
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This probe requires CUDA.")

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    model = DiffusionTransformer(
        n_blocks=args.blocks,
        n_heads=args.heads,
        c_a=args.c_a,
        c_s=args.c_s,
        c_z=args.c_z,
    ).to(device)
    model.eval()

    records = _random_inputs(args.lengths, args.c_a, args.c_s, args.c_z, device)
    a_batch, s_batch, z_batch, token_mask = _padded_inputs(
        records, args.c_a, args.c_s, args.c_z
    )
    z_records_cf = [_channel_first_z(z_i) for _, _, z_i in records]
    z_batch_cf = _channel_first_z(z_batch)

    @torch.no_grad()
    def run_sequential() -> list[torch.Tensor]:
        outputs = []
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for (a_i, s_i, _), z_i_cf in zip(records, z_records_cf):
                outputs.append(
                    model(
                        a_i,
                        s_i,
                        z_i_cf,
                        enable_efficient_fusion=True,
                    )
                )
        return outputs

    @torch.no_grad()
    def run_padded() -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(
                a_batch,
                s_batch,
                z_batch_cf,
                enable_efficient_fusion=True,
                token_mask=token_mask,
            )

    seq_times = _event_time_ms(lambda: run_sequential(), args.warmup, args.iters)
    pad_times = _event_time_ms(lambda: run_padded(), args.warmup, args.iters)

    seq_out = run_sequential()
    pad_out = run_padded()
    max_abs = 0.0
    for idx, out_i in enumerate(seq_out):
        n_token = out_i.shape[-2]
        diff = (out_i - pad_out[idx, :n_token]).float().abs().max().item()
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
