from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class StructureSection(BaseModel):
    path: str | None = None


class BasisSection(BaseModel):
    ecutwfc: float
    ecutrho: float | None = None


class KPointsSection(BaseModel):
    mode: Literal["gamma", "monkhorst-pack", "explicit"] = "gamma"
    grid: tuple[int, int, int] | None = None
    shift: tuple[float, float, float] | None = None


class XCSection(BaseModel):
    functional: Literal["lda", "pbe", "hf"] = "pbe"


class OccupationsSection(BaseModel):
    smearing: Literal["fixed", "fermi-dirac", "gaussian"] = "fixed"
    degauss: float | None = None


class SCFSection(BaseModel):
    max_iter: int = 60
    etol: float = 1e-8
    rhotol: float = 1e-6
    mixing_beta: float = 0.4
    mixing_ndim: int = 8
    kerker_q0: float = 1.0


class DiagSection(BaseModel):
    method: Literal["blocked_davidson"] = "blocked_davidson"
    nbands: int = 8
    block_size: int = 4
    max_subspace: int = 40
    residual_tol: float = 1e-8


class AutodiffSection(BaseModel):
    scf_diff: Literal["implicit", "unrolled", "stop_gradient"] = "implicit"
    stress_mode: Literal["strain_ad"] = "strain_ad"


class RuntimeSection(BaseModel):
    precision: Literal["float64", "float32"] = "float64"
    device: Literal["cpu", "gpu"] = "cpu"
    jit: bool = True


class SystemSection(BaseModel):
    charge: float = 0.0


class InputParams(BaseModel):
    structure: StructureSection = Field(default_factory=StructureSection)
    system: SystemSection = Field(default_factory=SystemSection)
    pseudopotentials: dict[str, str] = Field(default_factory=dict)
    basis: BasisSection
    kpoints: KPointsSection = Field(default_factory=KPointsSection)
    xc: XCSection = Field(default_factory=XCSection)
    occupations: OccupationsSection = Field(default_factory=OccupationsSection)
    scf: SCFSection = Field(default_factory=SCFSection)
    diagonalization: DiagSection = Field(default_factory=DiagSection)
    autodiff: AutodiffSection = Field(default_factory=AutodiffSection)
    runtime: RuntimeSection = Field(default_factory=RuntimeSection)

    @model_validator(mode="after")
    def _validate_cutoffs(self):
        if self.basis.ecutrho is None:
            object.__setattr__(self.basis, "ecutrho", 8.0 * self.basis.ecutwfc)
        if self.basis.ecutrho < 4.0 * self.basis.ecutwfc:
            raise ValueError("ecutrho must be >= 4 * ecutwfc")
        if self.kpoints.mode == "monkhorst-pack" and self.kpoints.grid is None:
            raise ValueError("kpoints.grid required for monkhorst-pack mode")
        return self


def load_input(path: str | Path) -> InputParams:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return InputParams.model_validate(data)
