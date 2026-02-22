"""UPF parser scaffold.

Target format: UPF v2.x. This module should parse XML content and return
PseudopotentialData objects plus structured radial/projector data.
"""

from __future__ import annotations
from pathlib import Path
from xml.etree import ElementTree as ET

from jaxpw.core.types import PseudopotentialData


def parse_upf(path: str | Path) -> PseudopotentialData:
    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()
    header = root.find("PP_HEADER")
    if header is None:
        raise ValueError(f"No PP_HEADER in UPF file: {path}")

    symbol = header.attrib.get("element", path.stem)
    z_val = float(header.attrib.get("z_valence", "0.0"))
    is_ultrasoft = header.attrib.get("is_ultrasoft", "F") == "T"
    is_paw = header.attrib.get("is_paw", "F") == "T"
    pp_type = "PAW" if is_paw else ("USPP" if is_ultrasoft else "NC")

    return PseudopotentialData(symbol=symbol, pp_type=pp_type, z_valence=z_val, raw={"path": str(path)})
