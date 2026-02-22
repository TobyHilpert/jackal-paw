"""UPF v2 parser utilities.

The parser extracts a practical subset of data needed by the current codebase:
- header metadata (`symbol`, `z_valence`, `pp_type`)
- radial mesh and local potential tables
- non-local projector tables (`beta` projectors and `D_ij` matrix)
- augmentation matrices (`Q_ij`) when available (USPP/PAW)

All parsed structured payload is exposed via `PseudopotentialData.raw`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np

from jackal.core.types import PseudopotentialData


def _parse_bool_flag(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().upper() in {"T", "TRUE", "1", "YES"}


def _parse_float_block(text: str | None) -> np.ndarray:
    if text is None:
        return np.array([], dtype=float)
    values = np.fromstring(text, sep=" ", dtype=float)
    return values if values.size else np.array([], dtype=float)


def _find_child_text(elem: ET.Element, name: str) -> str | None:
    child = elem.find(name)
    if child is None:
        return None
    return child.text


def _parse_header(root: ET.Element, path: Path) -> tuple[str, float, str, dict[str, Any]]:
    header = root.find("PP_HEADER")
    if header is None:
        raise ValueError(f"No PP_HEADER in UPF file: {path}")

    symbol = header.attrib.get("element", path.stem)
    z_val = float(header.attrib.get("z_valence", "0.0"))
    is_ultrasoft = _parse_bool_flag(header.attrib.get("is_ultrasoft"))
    is_paw = _parse_bool_flag(header.attrib.get("is_paw"))
    pp_type = "PAW" if is_paw else ("USPP" if is_ultrasoft else "NC")

    meta: dict[str, Any] = {
        "mesh_size": int(header.attrib["mesh_size"]) if "mesh_size" in header.attrib else None,
        "number_of_proj": int(header.attrib["number_of_proj"]) if "number_of_proj" in header.attrib else None,
        "functional": header.attrib.get("functional"),
    }
    return symbol, z_val, pp_type, meta


def _parse_mesh(root: ET.Element) -> dict[str, Any]:
    mesh = root.find("PP_MESH")
    if mesh is None:
        return {}

    r = _parse_float_block(_find_child_text(mesh, "PP_R"))
    rab = _parse_float_block(_find_child_text(mesh, "PP_RAB"))
    return {"r": r, "rab": rab}


def _parse_nonlocal(root: ET.Element) -> dict[str, Any]:
    nonlocal_node = root.find("PP_NONLOCAL")
    if nonlocal_node is None:
        return {}

    beta_projectors: list[np.ndarray] = []
    beta_l: list[int | None] = []
    for child in nonlocal_node:
        if not child.tag.startswith("PP_BETA"):
            continue
        beta_projectors.append(_parse_float_block(child.text))
        l_attr = child.attrib.get("angular_momentum")
        beta_l.append(int(l_attr) if l_attr is not None else None)

    dij_raw = _parse_float_block(_find_child_text(nonlocal_node, "PP_DIJ"))
    nproj = len(beta_projectors)
    dij = dij_raw.reshape((nproj, nproj)) if nproj and dij_raw.size == nproj * nproj else dij_raw

    qij_raw = _parse_float_block(_find_child_text(nonlocal_node, "PP_QIJ"))
    qij = qij_raw.reshape((nproj, nproj)) if nproj and qij_raw.size == nproj * nproj else qij_raw

    return {
        "beta_projectors": beta_projectors,
        "beta_angular_momentum": beta_l,
        "dij": dij,
        "qij": qij,
    }


def parse_upf(path: str | Path) -> PseudopotentialData:
    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()

    symbol, z_val, pp_type, header_meta = _parse_header(root, path)
    mesh = _parse_mesh(root)
    nonlocal_data = _parse_nonlocal(root)
    local_v = _parse_float_block(_find_child_text(root, "PP_LOCAL"))

    raw: dict[str, Any] = {
        "path": str(path),
        "header": header_meta,
        "mesh": mesh,
        "local_potential": local_v,
        "nonlocal": nonlocal_data,
    }

    return PseudopotentialData(symbol=symbol, pp_type=pp_type, z_valence=z_val, raw=raw)
