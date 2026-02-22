from __future__ import annotations

from jackal.xc import lda, pbe


def get_xc_functional(name: str):
    lname = name.lower()
    if lname == "lda":
        return lda
    if lname == "pbe":
        return pbe
    if lname == "hf":
        raise NotImplementedError("HF/EXX handled in electrostatics.exx (staged implementation)")
    raise ValueError(f"Unknown XC functional: {name}")
