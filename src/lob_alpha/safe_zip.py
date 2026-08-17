"""Safe manual extraction of the licensed Optiver training CSV."""

from __future__ import annotations

import re
import stat
import zipfile
from pathlib import Path, PurePosixPath

MAX_ZIP_MEMBERS = 10_000
MAX_TOTAL_UNCOMPRESSED_BYTES = 16 * 1024**3
MAX_TRAIN_CSV_BYTES = 8 * 1024**3
MAX_COMPRESSION_RATIO = 500.0
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def _normalized_member(member: zipfile.ZipInfo) -> str:
    if "\x00" in member.filename:
        raise ValueError("unsafe ZIP member contains a NUL byte")
    normalized = member.filename.replace("\\", "/")
    if normalized.startswith(("/", "//")) or re.match(r"^[A-Za-z]:", normalized):
        raise ValueError(f"unsafe ZIP absolute member path: {member.filename}")
    path = PurePosixPath(normalized)
    if not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe ZIP member path: {member.filename}")
    for component in path.parts:
        if component.endswith((" ", ".")) or ":" in component:
            raise ValueError(f"unsafe Windows ZIP member name: {member.filename}")
        stem = component.split(".", 1)[0].upper()
        if stem in _WINDOWS_RESERVED:
            raise ValueError(f"unsafe Windows reserved ZIP member name: {member.filename}")
    mode = member.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ValueError(f"unsafe ZIP symbolic link: {member.filename}")
    file_type = stat.S_IFMT(mode)
    if file_type and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
        raise ValueError(f"unsafe ZIP special file type: {member.filename}")
    return str(path)


def _validate_archive(bundle: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, str]]:
    members = bundle.infolist()
    if len(members) > MAX_ZIP_MEMBERS:
        raise ValueError(f"ZIP member count exceeds safety limit of {MAX_ZIP_MEMBERS}")
    total = sum(item.file_size for item in members)
    if total > MAX_TOTAL_UNCOMPRESSED_BYTES:
        raise ValueError("ZIP uncompressed size exceeds safety limit")
    validated: list[tuple[zipfile.ZipInfo, str]] = []
    destinations: dict[str, str] = {}
    for member in members:
        normalized = _normalized_member(member)
        destination_key = normalized.rstrip("/").casefold()
        if destination_key in destinations:
            raise ValueError(
                "ZIP contains duplicate or case-insensitive-colliding destinations: "
                f"{destinations[destination_key]!r} and {member.filename!r}"
            )
        destinations[destination_key] = member.filename
        compression_ratio = member.file_size / max(member.compress_size, 1)
        if member.file_size and compression_ratio > MAX_COMPRESSION_RATIO:
            raise ValueError(
                f"ZIP member compression ratio exceeds safety limit: {member.filename}"
            )
        validated.append((member, normalized))
    return validated


def extract_optiver_train_csv(
    archive_path: str | Path,
    *,
    output_path: str | Path = "data/raw/optiver/train.csv",
) -> Path:
    """Extract only train.csv, rejecting traversal, links, ambiguity and overwrite."""

    archive = Path(archive_path)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite licensed data: {output}")
    with zipfile.ZipFile(archive) as bundle:
        validated = _validate_archive(bundle)
        candidates = [
            item
            for item, normalized in validated
            if not item.is_dir()
            and PurePosixPath(normalized).name.casefold() == "train.csv"
        ]
        if len(candidates) != 1:
            raise ValueError(f"expected exactly one train.csv in archive, found {len(candidates)}")
        if candidates[0].file_size > MAX_TRAIN_CSV_BYTES:
            raise ValueError("train.csv uncompressed size exceeds safety limit")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".partial")
        if temporary.exists():
            raise FileExistsError(f"refusing to overwrite interrupted extraction: {temporary}")
        try:
            with bundle.open(candidates[0]) as source, temporary.open("xb") as target:
                copied = 0
                while chunk := source.read(1024 * 1024):
                    copied += len(chunk)
                    if copied > candidates[0].file_size or copied > MAX_TRAIN_CSV_BYTES:
                        raise ValueError("train.csv expanded beyond its declared safe size")
                    target.write(chunk)
            if copied != candidates[0].file_size:
                raise ValueError("train.csv extracted byte count does not match ZIP metadata")
            if temporary.stat().st_size == 0:
                raise ValueError("train.csv in the archive is empty")
            temporary.rename(output)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise
    if not output.is_file():
        raise OSError(f"expected extracted file does not exist: {output}")
    return output
