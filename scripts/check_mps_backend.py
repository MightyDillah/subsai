#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Manual runtime check for PyTorch MPS execution on Apple Silicon.

This script verifies that:
1) MPS backend is built and available.
2) A real tensor workload executes on MPS (not CPU).
3) Output numerically matches CPU output within tolerance.
"""

import argparse
import platform
import sys
import time

import torch


def _time_matmul(x: torch.Tensor, y: torch.Tensor, iterations: int, warmup: int) -> tuple[float, torch.Tensor]:
    is_mps = x.device.type == 'mps'

    out = None
    for _ in range(warmup):
        out = x @ y
    if is_mps:
        torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        out = x @ y
    if is_mps:
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed, out


def main() -> int:
    parser = argparse.ArgumentParser(description='Check whether PyTorch MPS is actually executing workloads.')
    parser.add_argument('--size', type=int, default=1024, help='Square matrix size for matmul workload.')
    parser.add_argument('--iterations', type=int, default=30, help='Timed iterations.')
    parser.add_argument('--warmup', type=int, default=5, help='Warmup iterations before timing.')
    parser.add_argument('--atol', type=float, default=1e-3, help='Absolute tolerance for CPU/MPS output comparison.')
    parser.add_argument('--rtol', type=float, default=1e-3, help='Relative tolerance for CPU/MPS output comparison.')
    args = parser.parse_args()

    print(f"python: {platform.python_version()}")
    print(f"platform: {platform.platform()}")
    print(f"machine: {platform.machine()}")
    print(f"torch: {torch.__version__}")

    has_mps_backend = hasattr(torch.backends, 'mps')
    mps_built = bool(has_mps_backend and torch.backends.mps.is_built())
    mps_available = bool(has_mps_backend and torch.backends.mps.is_available())
    print(f"mps_backend_present: {has_mps_backend}")
    print(f"mps_built: {mps_built}")
    print(f"mps_available: {mps_available}")

    if not mps_built or not mps_available:
        print("FAIL: MPS is not usable on this environment.")
        return 1

    torch.manual_seed(0)
    x_cpu = torch.randn((args.size, args.size), dtype=torch.float32, device='cpu')
    y_cpu = torch.randn((args.size, args.size), dtype=torch.float32, device='cpu')
    x_mps = x_cpu.to('mps')
    y_mps = y_cpu.to('mps')

    cpu_time, cpu_out = _time_matmul(x_cpu, y_cpu, args.iterations, args.warmup)
    mps_time, mps_out = _time_matmul(x_mps, y_mps, args.iterations, args.warmup)
    mps_out_cpu = mps_out.to('cpu')

    outputs_match = torch.allclose(cpu_out, mps_out_cpu, atol=args.atol, rtol=args.rtol)
    mps_executed = mps_out.device.type == 'mps'
    speed_ratio = cpu_time / mps_time if mps_time > 0 else float('inf')

    print(f"cpu_time_s: {cpu_time:.6f}")
    print(f"mps_time_s: {mps_time:.6f}")
    print(f"cpu_over_mps_speed_ratio: {speed_ratio:.3f}")
    print(f"mps_output_device: {mps_out.device}")
    print(f"cpu_mps_outputs_match: {outputs_match}")

    if not mps_executed:
        print("FAIL: Output tensor is not on MPS; workload did not run on MPS.")
        return 1
    if not outputs_match:
        print("FAIL: CPU and MPS outputs differ outside tolerance.")
        return 1

    print("PASS: MPS backend is executing workload and producing valid output.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
