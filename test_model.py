"""
test_model.py — Inference and training tests for Inf-Dehaze.

Run from the project root:
    python test_model.py
"""

import gc
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from inf_dehaze.model import InfDehaze

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
YAML_CFG = Path("./configs/dehaze_default.yaml")


def _reset_peak():
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)


def _peak_mb() -> float:
    if DEVICE.type == "cuda":
        return torch.cuda.max_memory_allocated(DEVICE) / 1e6
    return 0.0


def _current_mb() -> float:
    if DEVICE.type == "cuda":
        return torch.cuda.memory_allocated(DEVICE) / 1e6
    return 0.0


def _separator(title: str):
    width = 72
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _make_model(
    inference_batch_size: int = 4,
    use_residual_cache: bool = True,
    async_io: bool = True,
    dtype: torch.dtype = torch.float16,
    from_yaml: bool = False,
) -> InfDehaze:
    """Construct model, optionally from YAML, cast dtype, move to device."""
    if from_yaml and YAML_CFG.exists():
        model = InfDehaze.from_config(YAML_CFG)
        # Override inference knobs not easily changed post-construction
        model.inference_batch_size = inference_batch_size
        model.use_residual_cache   = use_residual_cache
        model.async_io             = async_io
    else:
        model = InfDehaze(
            inference_batch_size=inference_batch_size,
            use_residual_cache=use_residual_cache,
            async_io=async_io,
        )
    return model.to(dtype=dtype, device=DEVICE)


def _timed_run(model: nn.Module, x: torch.Tensor, label: str, n_repeats: int = 3):
    """Warm-up once then time n_repeats forward passes; return last output."""
    out = None
    # Warm-up
    with torch.no_grad():
        out = model(x)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    _reset_peak()
    times = []
    for i in range(n_repeats):
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(x)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  [{label}] iter {i + 1:02d}: {elapsed:.4f}s")

    avg = sum(times) / len(times)
    print(f"  [{label}] avg {avg:.4f}s  |  peak VRAM {_peak_mb():.1f} MB")
    return out


def _gc():
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Smoke test: single 256×256 crop (n_regions == 1)
# ─────────────────────────────────────────────────────────────────────────────

def test_single_crop():
    _separator("TEST 1 — Single 256×256 crop  (n_regions = 1)")
    torch.manual_seed(0)

    model = _make_model(inference_batch_size=1).eval()
    x = torch.zeros(1, 3, 256, 256, dtype=torch.float16, device=DEVICE)

    _reset_peak()
    with torch.no_grad():
        out = model(x)

    print(f"  input  : {tuple(x.shape)}")
    print(f"  output : {tuple(out.shape)}")
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    print(f"  peak VRAM : {_peak_mb():.1f} MB")
    print("  PASSED")
    _gc()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Multi-crop inference: 1024×1024  (4×4 = 16 tiles)
# ─────────────────────────────────────────────────────────────────────────────

def test_multi_crop_1k():
    _separator("TEST 2 — Multi-crop inference  1024×1024  (4×4 tiles, cache+async)")
    torch.manual_seed(0)

    model = _make_model(
        inference_batch_size=4,
        use_residual_cache=True,
        async_io=True,
    ).eval()

    x = torch.zeros(1, 3, 1024, 1024, dtype=torch.float16, device=DEVICE)
    out = _timed_run(model, x, label="1024 cache+async", n_repeats=3)

    assert out.shape == x.shape
    print(f"  output : {tuple(out.shape)}  PASSED")
    _gc()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Large-image inference: 8192×8192  (edge-deployment scenario)
# ─────────────────────────────────────────────────────────────────────────────

def test_large_image_8k():
    _separator("TEST 3 — Large-image inference  8192×8192  (32×32 tiles, batch=8)")
    torch.manual_seed(0)

    model = _make_model(
        inference_batch_size=4,
        use_residual_cache=True,
        async_io=True,
    ).eval()

    # 8192 / 256 = 32 → 1024 tiles; processed 8 at a time
    x = torch.zeros(1, 3, 8192, 8192, dtype=torch.float16, device=DEVICE)
    out = _timed_run(model, x, label="8192 cache+async", n_repeats=1)

    assert out.shape == x.shape
    print(f"  output : {tuple(out.shape)}  PASSED")
    del x, out
    _gc()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — Inference strategy comparison  (sync vs async vs no-cache)
#          on a medium 2048×2048 image
# ─────────────────────────────────────────────────────────────────────────────

def test_inference_strategy_comparison():
    _separator("TEST 4 — Inference strategy comparison  2048×2048")
    torch.manual_seed(0)

    x = torch.zeros(1, 3, 2048, 2048, dtype=torch.float16, device=DEVICE)

    strategies = [
        ("sync-cache",  dict(use_residual_cache=True,  async_io=False)),
        ("async-cache", dict(use_residual_cache=True,  async_io=True)),
        ("no-cache",    dict(use_residual_cache=False,  async_io=False)),
    ]

    results = {}
    for name, kwargs in strategies:
        model = _make_model(inference_batch_size=4, **kwargs).eval()
        _reset_peak()
        with torch.no_grad():
            _ = model(x)           # warm-up
        _reset_peak()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(x)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        vram = _peak_mb()
        results[name] = (elapsed, vram, out.shape)
        print(f"  {name:<14} : {elapsed:.4f}s  |  peak VRAM {vram:.1f} MB  |  out {tuple(out.shape)}")
        del model, out
        _gc()

    del x
    print("  PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — YAML config construction
# ─────────────────────────────────────────────────────────────────────────────

def test_from_yaml():
    _separator("TEST 5 — Build from YAML config")
    if not YAML_CFG.exists():
        print(f"  SKIPPED — {YAML_CFG} not found")
        return

    torch.manual_seed(0)
    model = _make_model(from_yaml=True, inference_batch_size=4).eval()

    x = torch.zeros(1, 3, 512, 512, dtype=torch.float16, device=DEVICE)
    with torch.no_grad():
        out = model(x)
    assert out.shape == x.shape
    print(f"  output : {tuple(out.shape)}  PASSED")
    _gc()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 — Training forward: single 256×256 crop
# ─────────────────────────────────────────────────────────────────────────────

def test_training_single_crop():
    _separator("TEST 6 — Training forward  256×256 (n_regions = 1)")
    torch.manual_seed(0)

    model = _make_model(inference_batch_size=1).train()
    # Use float32 for training stability
    model = model.float()
    x = torch.rand(1, 3, 256, 256, device=DEVICE)

    _reset_peak()
    output, moe_loss = model(x)

    print(f"  input    : {tuple(x.shape)}")
    print(f"  output   : {tuple(output.shape)}")
    print(f"  moe_loss : {moe_loss}")
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
    assert isinstance(moe_loss, (torch.Tensor, float))
    print(f"  peak VRAM : {_peak_mb():.1f} MB")
    print("  PASSED")
    _gc()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7 — Training forward: multi-crop 512×512 (2×2 = 4 tiles)
# ─────────────────────────────────────────────────────────────────────────────

def test_training_multi_crop():
    _separator("TEST 7 — Training forward  512×512  (2×2 tiles)")
    torch.manual_seed(0)

    model = _make_model(inference_batch_size=4).train().float()
    x = torch.rand(1, 3, 512, 512, device=DEVICE)

    _reset_peak()
    output, moe_loss = model(x)

    print(f"  input    : {tuple(x.shape)}")
    print(f"  output   : {tuple(output.shape)}")
    print(f"  moe_loss : {moe_loss}")
    assert output.shape == x.shape
    print(f"  peak VRAM : {_peak_mb():.1f} MB")
    print("  PASSED")
    _gc()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 8 — Backward pass + gradient check
# ─────────────────────────────────────────────────────────────────────────────

def test_backward():
    _separator("TEST 8 — Backward pass  256×256")
    torch.manual_seed(0)

    model = _make_model(inference_batch_size=1).train().float()
    x      = torch.rand(1, 3, 256, 256, device=DEVICE, requires_grad=False)
    target = torch.rand(1, 3, 256, 256, device=DEVICE)

    _reset_peak()
    output, moe_loss = model(x)

    # Pixel-level reconstruction loss + auxiliary MoE loss
    pixel_loss = nn.functional.l1_loss(output, target)
    total_loss = pixel_loss + 0.01 * moe_loss
    total_loss.backward()

    # Verify at least one parameter received a gradient
    grad_norms = [
        p.grad.norm().item()
        for p in model.parameters()
        if p.grad is not None
    ]
    assert len(grad_norms) > 0, "No parameters received gradients!"
    print(f"  pixel_loss : {pixel_loss.item():.6f}")
    print(f"  moe_loss   : {moe_loss if isinstance(moe_loss, float) else moe_loss.item():.6f}")
    print(f"  total_loss : {total_loss.item():.6f}")
    print(f"  params with grad : {len(grad_norms)} / {sum(1 for _ in model.parameters())}")
    print(f"  mean |grad| : {sum(grad_norms) / len(grad_norms):.6f}")
    print(f"  peak VRAM  : {_peak_mb():.1f} MB")
    print("  PASSED")
    _gc()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 9 — Optimizer step: verify weights update
# ─────────────────────────────────────────────────────────────────────────────

def test_optimizer_step():
    _separator("TEST 9 — Optimizer step  256×256")
    torch.manual_seed(0)

    model  = _make_model(inference_batch_size=1).train().float()
    optim  = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x      = torch.rand(1, 3, 256, 256, device=DEVICE)
    target = torch.rand_like(x)

    # Snapshot weights before step
    param_before = next(model.parameters()).detach().clone()

    optim.zero_grad()
    output, moe_loss = model(x)
    loss = nn.functional.l1_loss(output, target) + 0.01 * moe_loss
    loss.backward()
    optim.step()

    param_after = next(model.parameters()).detach().clone()
    changed = not torch.allclose(param_before, param_after)
    print(f"  loss           : {loss.item():.6f}")
    print(f"  weights updated: {changed}")
    assert changed, "Weights did not change after optimizer step!"
    print("  PASSED")
    _gc()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 10 — Train / eval mode switching + output consistency
# ─────────────────────────────────────────────────────────────────────────────

def test_mode_switching():
    _separator("TEST 10 — train() / eval() mode switching")
    torch.manual_seed(0)

    model = _make_model(inference_batch_size=2)
    x     = torch.rand(1, 3, 512, 512, device=DEVICE).half()

    # eval → inference (returns tensor)
    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    assert isinstance(out_eval, torch.Tensor), "eval() should return a Tensor"
    assert out_eval.shape == x.shape

    # train → training (returns tuple)
    model.train()
    model = model.float()
    x_f = x.float()
    out_train, moe_loss = model(x_f)
    assert isinstance(out_train, torch.Tensor), "train() should return (Tensor, loss)"
    assert out_train.shape == x_f.shape

    print(f"  eval  output : {tuple(out_eval.shape)}")
    print(f"  train output : {tuple(out_train.shape)},  moe_loss={moe_loss}")
    print("  PASSED")
    _gc()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 11 — Mixed-precision training with GradScaler
# ─────────────────────────────────────────────────────────────────────────────

def test_amp_training():
    _separator("TEST 11 — Mixed-precision training (AMP)  256×256")
    if DEVICE.type != "cuda":
        print("  SKIPPED — AMP requires CUDA")
        return

    torch.manual_seed(0)
    model  = _make_model(inference_batch_size=1, dtype=torch.float32).train()
    optim  = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    x      = torch.rand(1, 3, 256, 256, device=DEVICE)
    target = torch.rand_like(x)

    _reset_peak()
    optim.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output, moe_loss = model(x)
        loss = nn.functional.l1_loss(output, target) + 0.01 * moe_loss

    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()

    print(f"  loss       : {loss.item():.6f}")
    print(f"  output     : {tuple(output.shape)}")
    print(f"  peak VRAM  : {_peak_mb():.1f} MB")
    print("  PASSED")
    _gc()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 12 — Multi-step training loop (simulates real training)
# ─────────────────────────────────────────────────────────────────────────────

def test_training_loop():
    _separator("TEST 12 — Multi-step training loop  256×256  (5 steps)")
    torch.manual_seed(0)

    model = _make_model(inference_batch_size=1).train().float()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    n_steps   = 5
    crop_size = 256
    losses    = []

    _reset_peak()
    for step in range(n_steps):
        x      = torch.rand(1, 3, crop_size, crop_size, device=DEVICE)
        target = torch.rand_like(x)

        t0 = time.perf_counter()
        optim.zero_grad()
        output, moe_loss = model(x)
        loss = nn.functional.l1_loss(output, target) + 0.01 * moe_loss
        loss.backward()
        optim.step()
        elapsed = time.perf_counter() - t0

        losses.append(loss.item())
        print(f"  step {step + 1:02d}: loss={loss.item():.6f}  ({elapsed:.3f}s)")

    print(f"  final loss : {losses[-1]:.6f}")
    print(f"  peak VRAM  : {_peak_mb():.1f} MB")
    # Loss need not strictly decrease in 5 random steps, but should be finite
    assert all(l == l for l in losses), "NaN detected in losses!"  # NaN != NaN
    print("  PASSED")
    _gc()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 13 — Parameter count summary
# ─────────────────────────────────────────────────────────────────────────────

def test_param_count():
    _separator("TEST 13 — Parameter count")
    model = _make_model()

    total   = sum(p.numel() for p in model.parameters())
    enc     = sum(p.numel() for p in model.encoder.parameters())
    bn      = sum(p.numel() for p in model.bottleneck.parameters())
    dec     = sum(p.numel() for p in model.decoder.parameters())

    print(f"  Total       : {total:>12,}  ({total / 1e6:.2f} M)")
    print(f"  Encoder     : {enc:>12,}  ({enc / 1e6:.2f} M)")
    print(f"  Bottleneck  : {bn:>12,}  ({bn / 1e6:.2f} M)")
    print(f"  Decoder     : {dec:>12,}  ({dec / 1e6:.2f} M)")
    assert total == enc + bn + dec
    print("  PASSED")
    _gc()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nDevice : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(DEVICE)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(DEVICE).total_memory / 1e9:.1f} GB")

    # ── Inference tests ──────────────────────────────────────────────────────
    test_param_count()
    test_single_crop()
    test_from_yaml()
    test_multi_crop_1k()
    test_inference_strategy_comparison()

    # ── Large-image test (comment out if VRAM < 16 GB) ──────────────────────
    test_large_image_8k()

    # ── Training tests ───────────────────────────────────────────────────────
    test_training_single_crop()
    test_training_multi_crop()
    test_backward()
    test_optimizer_step()
    test_mode_switching()
    test_amp_training()
    test_training_loop()

    _separator("ALL TESTS COMPLETE")