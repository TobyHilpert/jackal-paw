"""Global runtime configuration helpers (JAX precision, device selection, etc.)."""

from __future__ import annotations


def configure_jax(enable_x64: bool = True) -> None:
    try:
        import jax
        jax.config.update("jax_enable_x64", enable_x64)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Failed to configure JAX") from exc
