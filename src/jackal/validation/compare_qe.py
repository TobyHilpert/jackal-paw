"""Utilities to compare jackal results against Quantum ESPRESSO reference outputs."""

from __future__ import annotations


def compare_scalar_results(result: dict[str, float], reference: dict[str, float], tolerances: dict[str, float]) -> dict:
    failures: list[dict[str, float | str]] = []
    for key, ref in reference.items():
        if key not in result:
            failures.append({"key": key, "reason": "missing"})
            continue
        tol = float(tolerances.get(key, 0.0))
        err = abs(float(result[key]) - float(ref))
        if err > tol:
            failures.append({"key": key, "error": err, "tolerance": tol})
    return {"ok": len(failures) == 0, "failures": failures}
