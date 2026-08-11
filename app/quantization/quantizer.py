"""Stage 8 — quantization orchestrator and injectable backends.

This module provides:

* ``Quantizer`` — a ``Protocol`` (same injectable-backend pattern as
  ``ModelBackend`` in Stage 4, ``EmbeddingBackend`` in Stage 2).
* ``MockQuantizer`` — deterministic, no-deps quantizer for tests.
* ``run_quantization_matrix`` — the Stage 8 orchestrator that iterates
  over methods × bit-widths and produces a ``QuantReport``.
* ``quantize_single`` — quantize once with a given method + bit-width.
* ``select_best_config`` — pick the best result given VRAM / size constraints.

Real quantizer implementations live in separate modules:
``export_gptq.py``, ``export_awq.py``, ``export_gguf.py`` — as specified
in the README repo layout.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Protocol, runtime_checkable

from app.quantization.config import (
    QuantConfig,
    estimate_model_size_gb,
    estimate_quality,
    estimate_tokens_per_sec,
    estimate_vram_gb,
)
from app.quantization.export_awq import AWQQuantizer
from app.quantization.export_gguf import GGUFQuantizer, gguf_type_to_bits

# Import the real quantizer implementations from their dedicated modules.
from app.quantization.export_gptq import GPTQQuantizer
from app.schemas.quantization import (
    QuantMethod,
    QuantReport,
    QuantResult,
    QuantStatus,
)

logger = logging.getLogger(__name__)

__all__ = [
    "Quantizer",
    "MockQuantizer",
    "GPTQQuantizer",
    "AWQQuantizer",
    "GGUFQuantizer",
    "QuantResult",
    "QuantStatus",
    "QuantMethod",
    "QuantConfig",
    "quantize_single",
    "select_best_config",
    "run_quantization_matrix",
]


# ---------------------------------------------------------------------------
# Quantizer Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Quantizer(Protocol):
    """Anything that can quantize a checkpoint and return a ``QuantResult``.

    Follows the same injectable-backend pattern as ``ModelBackend``
    (Stage 4/6/7) and ``EmbeddingBackend`` (Stage 2): tests inject
    ``MockQuantizer``, production code uses the real implementation.
    """

    def quantize(
        self,
        source_checkpoint: str,
        output_path: str,
        bit_width: int | None = None,
    ) -> QuantResult:
        """Quantize *source_checkpoint* and write the result to *output_path*.

        Parameters
        ----------
        source_checkpoint:
            Path / URI to the full-precision checkpoint (from Stage 5).
        output_path:
            Where to write the quantized artifact.
        bit_width:
            Target precision. ``None`` means "use method default".
        """
        ...


# ---------------------------------------------------------------------------
# MockQuantizer — fully deterministic, no external deps
# ---------------------------------------------------------------------------


class MockQuantizer:
    """Deterministic quantizer for testing.

    Returns a canned ``QuantResult`` with heuristic-based values derived
    from the bit-width. The ``results`` dict is keyed by
    ``f"{method.value}:{bit_width}"`` so individual configs can be
    overridden per test.
    """

    def __init__(
        self,
        results: dict[str, QuantResult] | None = None,
        default_method: QuantMethod = QuantMethod.NONE,
        default_bit_width: int = 4,
    ):
        self._results = results or {}
        self._method = default_method
        self._bits = default_bit_width
        self.call_count = 0
        self.last_call: dict | None = None

    def quantize(
        self,
        source_checkpoint: str,
        output_path: str,
        bit_width: int | None = None,
    ) -> QuantResult:
        self.call_count += 1
        self.last_call = {
            "source": source_checkpoint,
            "output": output_path,
            "bits": bit_width,
        }
        bits = bit_width if bit_width is not None else self._bits
        key = f"{self._method.value}:{bits}"

        if key in self._results:
            return self._results[key]

        # Heuristic defaults based on the method + bits.
        est_vram = estimate_vram_gb(self._method, bits)
        est_size = estimate_model_size_gb(self._method, bits)
        est_quality = estimate_quality(self._method, bits)
        tps = estimate_tokens_per_sec(self._method, bits)

        return QuantResult(
            quant_method=self._method,
            bit_width=bits,
            quantized_model_size_gb=est_size,
            estimated_vram_gb=est_vram,
            measured_vram_gb=None,  # mock can't measure
            tokens_per_sec=tps,
            model_cwe_macro_f1=est_quality,
            exec_pass_rate=round(est_quality, 2),
            status=QuantStatus.COMPLETED,
            checkpoint_path=output_path,
            notes=f"mock {self._method.value} @ {bits}-bit",
        )


# ---------------------------------------------------------------------------
# NoOpQuantizer — pass-through for "no quantization" baseline
# ---------------------------------------------------------------------------


class _NoOpQuantizer:
    """Pass-through quantizer that records the source checkpoint as-is.

    Used when ``QuantMethod.NONE`` is selected — produces a baseline
    (unquantized) result for comparison.
    """

    def quantize(
        self,
        source_checkpoint: str,
        output_path: str,
        bit_width: int | None = None,
    ) -> QuantResult:
        size_gb = _estimate_unquantized_size(source_checkpoint)
        return QuantResult(
            quant_method=QuantMethod.NONE,
            bit_width=16,
            quantized_model_size_gb=size_gb,
            measured_vram_gb=None,
            estimated_vram_gb=15.0,
            tokens_per_sec=None,
            model_cwe_macro_f1=None,
            exec_pass_rate=None,
            status=QuantStatus.COMPLETED,
            checkpoint_path=output_path,
            notes="no quantization (FP16 baseline)",
        )


def _estimate_unquantized_size(checkpoint_path: str) -> float:
    """Roughly estimate the size of an unquantized checkpoint directory (GB)."""
    total = 0
    p = Path(checkpoint_path)
    if p.is_dir():
        for f in p.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    elif p.is_file():
        total = p.stat().st_size
    return round(total / (1024 ** 3), 2)  # bytes → GB


# ---------------------------------------------------------------------------
# Quantizer factory
# ---------------------------------------------------------------------------


def _make_quantizer(method: QuantMethod, config: QuantConfig) -> Quantizer:
    """Build the appropriate real quantizer for *method*."""
    if method == QuantMethod.GPTQ:
        return GPTQQuantizer(config=config.gptq_config)
    if method == QuantMethod.AWQ:
        return AWQQuantizer(config=config.awq_config)
    if method == QuantMethod.GGUF:
        return GGUFQuantizer(config=config.gguf_config)
    if method == QuantMethod.NONE:
        return _NoOpQuantizer()
    raise ValueError(f"Unknown quantization method: {method}")


# ---------------------------------------------------------------------------
# Per-method quantization helper
# ---------------------------------------------------------------------------


def quantize_single(
    method: QuantMethod,
    bit_width: int,
    config: QuantConfig,
) -> QuantResult:
    """Quantize the checkpoint once with the given method + bit-width.

    In ``dry_run`` or ``mock`` mode, returns heuristic estimates without
    calling any external library.
    """
    output_path = os.path.join(
        config.output_base,
        f"{method.value}_bits{bit_width}",
    )

    if config.mock:
        quantizer = MockQuantizer(
            default_method=method,
            default_bit_width=bit_width,
        )
        return quantizer.quantize(
            source_checkpoint=config.source_checkpoint,
            output_path=output_path,
            bit_width=bit_width,
        )

    if config.dry_run:
        est_vram = estimate_vram_gb(method, bit_width)
        est_size = estimate_model_size_gb(method, bit_width)
        est_quality = estimate_quality(method, bit_width)
        tps = estimate_tokens_per_sec(method, bit_width)
        return QuantResult(
            quant_method=method,
            bit_width=bit_width,
            quantized_model_size_gb=est_size,
            estimated_vram_gb=est_vram,
            measured_vram_gb=None,
            tokens_per_sec=tps,
            model_cwe_macro_f1=est_quality,
            exec_pass_rate=round(est_quality, 2),
            status=QuantStatus.COMPLETED,
            checkpoint_path=output_path,
            notes=f"dry-run {method.value} @ {bit_width}-bit",
        )

    # Real quantization.
    quantizer = _make_quantizer(method, config)
    return quantizer.quantize(
        source_checkpoint=config.source_checkpoint,
        output_path=output_path,
        bit_width=bit_width,
    )


# ---------------------------------------------------------------------------
# Best-config selection
# ---------------------------------------------------------------------------

# Weighting for the quality/size/speed trade-off score (sums to 1.0).
# Higher = more quality, less size, less speed emphasis.
_QUALITY_WEIGHT = 0.6
_SIZE_WEIGHT = 0.2
_SPEED_WEIGHT = 0.2


def score_quality_size_speed(result: QuantResult) -> float:
    """Compute a normalized 0–1 score for quality vs. size and speed.

    Quality dominates (0.6 weight); size and speed are normalized against
    the full-precision baseline (14 GB, ~30 t/s on a 7B GPU model).
    Returns ``0.0`` for failed results.
    """
    if result.status == QuantStatus.FAILED:
        return 0.0

    # Quality: model_cwe_macro_f1 or estimated fallback.
    quality = result.model_cwe_macro_f1
    if quality is None:
        quality = estimate_quality(result.quant_method, result.bit_width or 4)

    # Size: lower is better. Normalize against FP16 baseline (14 GB).
    size_gb = result.quantized_model_size_gb
    size_score = max(0.0, 1.0 - (size_gb / 14.0))

    # Speed: higher is better. Normalize against a 7B GPU baseline of ~30 t/s.
    tps = result.tokens_per_sec
    if tps is not None:
        speed_score = min(1.0, tps / 30.0)
    else:
        speed_score = 0.5  # unknown — neutral

    return (
        _QUALITY_WEIGHT * quality
        + _SIZE_WEIGHT * size_score
        + _SPEED_WEIGHT * speed_score
    )


def select_best_config(
    results: list[QuantResult],
    *,
    target_vram_gb: float | None = None,
    target_size_gb: float | None = None,
) -> QuantResult | None:
    """Select the best quantization configuration from a list of results.

    If ``target_vram_gb`` or ``target_size_gb`` is provided, only results
    that fit within the budget are considered. Among the survivors, the
    one with the highest ``score_quality_size_speed`` wins.

    Returns ``None`` if no result passes the filters.
    """
    candidates = [
        r for r in results
        if r.status == QuantStatus.COMPLETED
    ]

    if target_vram_gb is not None:
        candidates = [
            r for r in candidates
            if r.estimated_vram_gb <= target_vram_gb
        ]

    if target_size_gb is not None:
        candidates = [
            r for r in candidates
            if r.quantized_model_size_gb <= target_size_gb
        ]

    if not candidates:
        return None

    return max(candidates, key=score_quality_size_speed)


# ---------------------------------------------------------------------------
# Full matrix runner
# ---------------------------------------------------------------------------


def run_quantization_matrix(config: QuantConfig) -> QuantReport:
    """Run the full quantization matrix and return a ``QuantReport``.

    Iterates over all combinations of ``config.methods`` ×
    ``config.bit_widths`` and produces a ``QuantReport`` with a best-config
    recommendation.

    * **Mock mode** (``config.mock=True``): no external libraries called.
    * **Dry-run mode** (``config.dry_run=True``): heuristic estimates only.
    * **Real mode**: each method×bit-width combination is attempted;
      failures produce a ``FAILED`` result with an error message rather
      than crashing the whole matrix.
    """
    start_time = time.time()
    run_id = f"stage8-{uuid.uuid4().hex[:8]}"

    results: list[QuantResult] = []

    for method in config.methods:
        if method == QuantMethod.GGUF:
            # GGUF iterates over its configured quant_types.
            for qt in config.gguf_config.quant_types:
                try:
                    bits = gguf_type_to_bits(qt)
                except ValueError:
                    continue
                result = _try_quantize(method, bits, config)
                results.append(result)
        else:
            for bits in config.bit_widths:
                result = _try_quantize(method, bits, config)
                results.append(result)

    best = select_best_config(
        results,
        target_vram_gb=config.target_vram_gb,
        target_size_gb=config.target_size_gb,
    )

    elapsed = time.time() - start_time
    manifest = {
        "run_id": run_id,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
        "elapsed_seconds": round(elapsed, 2),
        "base_model": config.base_model,
        "source_checkpoint": config.source_checkpoint,
        "methods": [m.value for m in config.methods],
        "bit_widths": config.bit_widths,
        "dry_run": config.dry_run,
        "mock": config.mock,
    }

    report = QuantReport(
        run_id=run_id,
        base_model=config.base_model,
        source_checkpoint=config.source_checkpoint,
        results=results,
        best_result=best,
        manifest=manifest,
    )

    logger.info(
        "Stage 8 complete: %d results, best=%s",
        len(results),
        best.quant_method if best else "none",
    )

    return report


def _try_quantize(
    method: QuantMethod,
    bits: int,
    config: QuantConfig,
) -> QuantResult:
    """Attempt a single quantization, catching errors into a FAILED result."""
    try:
        return quantize_single(method, bits, config)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Quantize %s @ %d-bit failed: %s", method.value, bits, exc,
        )
        return QuantResult(
            quant_method=method,
            bit_width=bits,
            quantized_model_size_gb=0.0,
            estimated_vram_gb=0.0,
            measured_vram_gb=None,
            tokens_per_sec=None,
            model_cwe_macro_f1=None,
            exec_pass_rate=None,
            status=QuantStatus.FAILED,
            checkpoint_path="",
            error=str(exc),
            notes=f"failed {method.value} @ {bits}-bit",
        )
