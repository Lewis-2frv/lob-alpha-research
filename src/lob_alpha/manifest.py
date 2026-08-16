"""Content-addressed data and experiment manifests."""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, default=str, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Real) and not isinstance(value, bool) and not math.isfinite(float(value)):
        return None
    return value


def build_run_manifest(
    *,
    config_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    session_date: str,
    rows: int,
    quality: dict[str, Any],
) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "config_path": str(Path(config_path)),
        "config_sha256": sha256_file(config_path),
        "input_path": str(Path(input_path)),
        "input_sha256": sha256_file(input_path),
        "output_path": str(Path(output_path)),
        "session_date": session_date,
        "processed_rows": rows,
        "quality": quality,
    }
