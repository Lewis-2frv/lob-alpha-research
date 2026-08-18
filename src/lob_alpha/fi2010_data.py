"""Source-bound, target-safe archive and matrix handling for FI-2010."""

from __future__ import annotations

import ctypes
import json
import os
import re
import shutil
import stat
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .fi2010_config import FI2010Config
from .manifest import sha256_file

MAX_ARCHIVE_MEMBERS = 20_000
MAX_TOTAL_UNCOMPRESSED_BYTES = 100 * 1024**3
MAX_MEMBER_UNCOMPRESSED_BYTES = 8 * 1024**3
MAX_COMPRESSION_RATIO = 1_000.0
COPY_BLOCK_BYTES = 4 * 1024**2
EXPECTED_ROWS = 149
FEATURE_ROWS = 144
LABEL_ROWS = 5
CLAIM_FILENAME = "FI2010_CF9_HOLDOUT_CLAIMED.lock"
SEAL_FILENAME = "FI2010_CF9_HOLDOUT_STARTED.json"
ANCHOR_FILENAME = "FI2010_CF9_HOLDOUT_COMPLETE_ANCHOR.json"
_TRAIN_NAME = "Train_Dst_NoAuction_ZScore_CF_{fold}.txt"
_TEST_NAME = "Test_Dst_NoAuction_ZScore_CF_{fold}.txt"
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


@dataclass(frozen=True)
class FI2010Matrix:
    """One publisher-provided snapshot matrix transposed to observations by fields."""

    features: np.ndarray
    labels: np.ndarray
    member: str

    def primary_target(self, primary_label_row: int = 4) -> np.ndarray:
        return self.labels[:, primary_label_row - 1]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Durably write sorted JSON by same-directory atomic promotion."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(f"interrupted JSON write requires review: {temporary}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return output


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalized_member(member: zipfile.ZipInfo) -> str:
    filename = member.filename
    if "\x00" in filename:
        raise ValueError("unsafe ZIP member contains a NUL byte")
    if "\\" in filename:
        raise ValueError(f"unsafe ZIP member uses backslashes: {filename!r}")
    if filename.startswith(("/", "//")) or re.match(r"^[A-Za-z]:", filename):
        raise ValueError(f"unsafe absolute ZIP member: {filename!r}")
    path = PurePosixPath(filename)
    if not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe ZIP member path: {filename!r}")
    for component in path.parts:
        if component.endswith((" ", ".")) or ":" in component:
            raise ValueError(f"unsafe Windows ZIP member name: {filename!r}")
        if component.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
            raise ValueError(f"unsafe Windows reserved ZIP member: {filename!r}")
    mode = member.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ValueError(f"unsafe ZIP symbolic link: {filename!r}")
    file_type = stat.S_IFMT(mode)
    if file_type and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
        raise ValueError(f"unsafe ZIP special member: {filename!r}")
    if member.flag_bits & 0x1:
        raise ValueError(f"encrypted ZIP members are not supported: {filename!r}")
    return str(path)


def validate_zip_directory(bundle: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """Validate central-directory metadata without opening any payload member."""

    members = bundle.infolist()
    if not members or len(members) > MAX_ARCHIVE_MEMBERS:
        raise ValueError("ZIP member count is empty or exceeds the registered safety limit")
    total = 0
    validated: dict[str, zipfile.ZipInfo] = {}
    destinations: dict[str, str] = {}
    for member in members:
        normalized = _normalized_member(member)
        key = normalized.rstrip("/").casefold()
        if key in destinations:
            raise ValueError(
                "duplicate or case-insensitive-colliding ZIP destinations: "
                f"{destinations[key]!r} and {member.filename!r}"
            )
        destinations[key] = member.filename
        if member.file_size < 0 or member.file_size > MAX_MEMBER_UNCOMPRESSED_BYTES:
            raise ValueError(f"ZIP member exceeds the per-member size limit: {member.filename}")
        total += member.file_size
        if total > MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise ValueError("ZIP total uncompressed size exceeds the safety limit")
        ratio = member.file_size / max(member.compress_size, 1)
        if member.file_size and ratio > MAX_COMPRESSION_RATIO:
            raise ValueError(f"ZIP compression ratio exceeds the safety limit: {member.filename}")
        validated[normalized] = member
    return validated


def available_memory_bytes() -> int | None:
    """Return currently available physical bytes on Windows, if discoverable."""

    if os.name != "nt":
        try:
            page = os.sysconf("SC_PAGE_SIZE")
            pages = os.sysconf("SC_AVPHYS_PAGES")
        except (AttributeError, OSError, ValueError):
            return None
        return int(page * pages)

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.available_physical)


def preflight_resources(destination: str | Path, required_disk_bytes: int) -> dict[str, int | None]:
    """Fail clearly before extraction or parsing when local resources are insufficient."""

    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    free_disk = shutil.disk_usage(destination_path).free
    available_memory = available_memory_bytes()
    disk_floor = required_disk_bytes + 256 * 1024**2
    if free_disk < disk_floor:
        raise OSError(f"insufficient disk: need at least {disk_floor}, available {free_disk}")
    if available_memory is not None and available_memory < 768 * 1024**2:
        raise MemoryError(
            "FI-2010 parsing requires at least 768 MiB of currently available physical memory"
        )
    return {"free_disk_bytes": free_disk, "available_memory_bytes": available_memory}


def verify_outer_archive(config: FI2010Config, archive_path: str | Path) -> dict[str, Any]:
    """Verify exact source identity, then validate outer metadata without extraction."""

    archive = Path(archive_path).resolve()
    actual_size = archive.stat().st_size
    if actual_size != config.source.outer_archive_size:
        raise ValueError(
            f"outer archive size mismatch: expected {config.source.outer_archive_size}, "
            f"found {actual_size}"
        )
    actual_sha256 = sha256_file(archive)
    if actual_sha256 != config.source.outer_archive_sha256:
        raise ValueError("outer archive SHA-256 does not match the registered FI-2010 source")
    with zipfile.ZipFile(archive) as bundle:
        members = validate_zip_directory(bundle)
        if set(members) != {config.source.inner_member}:
            raise ValueError("outer archive must contain exactly the registered inner ZIP member")
        inner = members[config.source.inner_member]
        if inner.is_dir() or inner.file_size == 0:
            raise ValueError("registered inner archive member is missing or empty")
        inner_identity = zip_member_identity(inner)
    return {
        "archive_path": str(archive),
        "archive_size": actual_size,
        "archive_sha256": actual_sha256,
        "inner_member": inner_identity,
        "dataset_id": config.source.dataset_id,
        "pid": config.source.pid,
        "title": config.source.title,
        "licence": config.source.licence,
        "paper_doi": config.source.paper_doi,
        "network_access": False,
    }


def import_inner_archive(
    config: FI2010Config,
    archive_path: str | Path,
    *,
    prepared_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Extract only the verified inner ZIP using partial-plus-atomic promotion."""

    verified = verify_outer_archive(config, archive_path)
    root = Path(prepared_dir or config.data.prepared_dir).resolve()
    resources = preflight_resources(root, int(verified["inner_member"]["file_size"]))
    output = root / "BenchmarkDatasets.zip"
    manifest_path = root / "source_manifest.json"
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or manifest_path.exists() or partial.exists():
        if output.is_file() and manifest_path.is_file() and not partial.exists():
            existing = read_json(manifest_path)
            if (
                existing.get("outer_archive_sha256") == verified["archive_sha256"]
                and existing.get("inner_archive_sha256") == sha256_file(output)
                and existing.get("inner_archive_size") == output.stat().st_size
            ):
                return existing
        raise FileExistsError("prepared FI-2010 source is incomplete, altered, or unmanifested")
    copied = 0
    try:
        with zipfile.ZipFile(verified["archive_path"]) as bundle:
            members = validate_zip_directory(bundle)
            member = members[config.source.inner_member]
            with bundle.open(member, "r") as source, partial.open("xb") as target:
                while chunk := source.read(COPY_BLOCK_BYTES):
                    copied += len(chunk)
                    if copied > member.file_size:
                        raise ValueError("inner archive expanded beyond declared ZIP metadata")
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
        if copied != int(verified["inner_member"]["file_size"]):
            raise ValueError("inner archive byte count does not match ZIP metadata")
        os.replace(partial, output)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    payload = {
        "stage": "fi2010_source_import",
        "created_utc": utc_now(),
        "claim_eligible": not config.synthetic,
        "outer_archive_path": verified["archive_path"],
        "outer_archive_size": verified["archive_size"],
        "outer_archive_sha256": verified["archive_sha256"],
        "outer_member": verified["inner_member"],
        "inner_archive_path": str(output),
        "inner_archive_size": output.stat().st_size,
        "inner_archive_sha256": sha256_file(output),
        "source_attribution": {
            "dataset_id": config.source.dataset_id,
            "pid": config.source.pid,
            "title": config.source.title,
            "licence": config.source.licence,
            "paper_doi": config.source.paper_doi,
        },
        "resource_preflight": resources,
        "network_access": False,
    }
    atomic_json(manifest_path, payload)
    return payload


def zip_member_identity(member: zipfile.ZipInfo) -> dict[str, Any]:
    return {
        "path": _normalized_member(member),
        "file_size": member.file_size,
        "compressed_size": member.compress_size,
        "crc32": f"{member.CRC:08x}",
        "compression": member.compress_type,
    }


def _expected_member(
    members: dict[str, zipfile.ZipInfo], config: FI2010Config, *, split: str, fold: int
) -> zipfile.ZipInfo:
    basename = (_TRAIN_NAME if split == "train" else _TEST_NAME).format(fold=fold)
    representation = f"/{config.data.representation}/"
    candidates = [
        member
        for path, member in members.items()
        if not member.is_dir() and path.endswith(f"/{basename}") and representation in f"/{path}"
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one {split} CF_{fold} member in the primary representation, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def is_cf9_test_member(path: str) -> bool:
    return PurePosixPath(path).name == _TEST_NAME.format(fold=9)


def parse_fi2010_member(
    bundle: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    *,
    allow_cf9_test: bool = False,
) -> FI2010Matrix:
    """Parse one 149-row member as float32; payload access is guarded before open."""

    normalized = _normalized_member(member)
    if is_cf9_test_member(normalized) and not allow_cf9_test:
        raise PermissionError("CF_9 test payload access is sealed outside the holdout stage")
    if member.file_size == 0 or member.file_size > MAX_MEMBER_UNCOMPRESSED_BYTES:
        raise ValueError("FI-2010 member is empty or exceeds the parser safety bound")
    available = available_memory_bytes()
    if available is not None and available < 768 * 1024**2:
        raise MemoryError("insufficient available memory for bounded FI-2010 float32 parsing")
    with bundle.open(member, "r") as handle:
        matrix = np.loadtxt(handle, dtype=np.float32, ndmin=2)
    if matrix.ndim != 2 or matrix.shape[0] != EXPECTED_ROWS or matrix.shape[1] == 0:
        raise ValueError(f"FI-2010 member must have exactly 149 rows: {normalized} {matrix.shape}")
    features = np.ascontiguousarray(matrix[:FEATURE_ROWS, :].T)
    labels_float = matrix[FEATURE_ROWS:, :].T
    if not np.isfinite(features).all() or not np.isfinite(labels_float).all():
        raise ValueError(f"FI-2010 member contains NaN or infinity: {normalized}")
    if not np.equal(labels_float, np.rint(labels_float)).all():
        raise ValueError(f"FI-2010 labels must be integral: {normalized}")
    labels = np.ascontiguousarray(labels_float.astype(np.int8))
    if not np.isin(labels, (1, 2, 3)).all():
        raise ValueError(f"FI-2010 labels must be in {{1,2,3}}: {normalized}")
    del matrix, labels_float
    return FI2010Matrix(features=features, labels=labels, member=normalized)


def verify_prepared_source(
    config: FI2010Config, prepared_dir: str | Path | None = None
) -> tuple[Path, dict[str, Any]]:
    root = Path(prepared_dir or config.data.prepared_dir).resolve()
    source_path = root / "source_manifest.json"
    source = read_json(source_path)
    if source.get("stage") != "fi2010_source_import":
        raise ValueError("not an FI-2010 source import manifest")
    if source.get("outer_archive_size") != config.source.outer_archive_size:
        raise ValueError("prepared source outer-archive size binding changed")
    if source.get("source_attribution") != {
        "dataset_id": config.source.dataset_id,
        "pid": config.source.pid,
        "title": config.source.title,
        "licence": config.source.licence,
        "paper_doi": config.source.paper_doi,
    }:
        raise ValueError("prepared source attribution binding changed")
    inner = Path(source["inner_archive_path"])
    if inner.resolve() != (root / "BenchmarkDatasets.zip").resolve():
        raise ValueError("source manifest points outside the prepared FI-2010 directory")
    if inner.stat().st_size != source["inner_archive_size"]:
        raise ValueError("prepared inner archive size changed")
    if sha256_file(inner) != source["inner_archive_sha256"]:
        raise ValueError("prepared inner archive SHA-256 changed")
    if source["outer_archive_sha256"] != config.source.outer_archive_sha256:
        raise ValueError("configuration source identity does not match the imported archive")
    return inner, source


def audit_inner_archive(
    config: FI2010Config,
    *,
    prepared_dir: str | Path | None = None,
    validate_payloads: bool = True,
) -> dict[str, Any]:
    """Audit development and Train_CF_9 payloads, never opening Test_CF_9."""

    root = Path(prepared_dir or config.data.prepared_dir).resolve()
    for state_name in (CLAIM_FILENAME, SEAL_FILENAME, ANCHOR_FILENAME):
        if (root / state_name).exists():
            raise FileExistsError(
                "refusing to rewrite FI-2010 audit manifests after CF_9 holdout release began"
            )
    inner_path, source = verify_prepared_source(config, root)
    development_members: list[dict[str, Any]] = []
    opened_members: list[str] = []
    with zipfile.ZipFile(inner_path) as bundle:
        members = validate_zip_directory(bundle)
        for fold in config.data.development_folds:
            for split in ("train", "test"):
                member = _expected_member(members, config, split=split, fold=fold)
                identity = zip_member_identity(member)
                identity.update({"fold": fold, "split": split})
                if validate_payloads:
                    parsed = parse_fi2010_member(bundle, member)
                    identity["rows"] = EXPECTED_ROWS
                    identity["observations"] = int(parsed.features.shape[0])
                    opened_members.append(parsed.member)
                    del parsed
                development_members.append(identity)
        train9 = _expected_member(members, config, split="train", fold=config.data.final_fold)
        train9_identity = zip_member_identity(train9)
        train9_identity.update({"fold": config.data.final_fold, "split": "train"})
        if validate_payloads:
            parsed = parse_fi2010_member(bundle, train9)
            train9_identity["rows"] = EXPECTED_ROWS
            train9_identity["observations"] = int(parsed.features.shape[0])
            opened_members.append(parsed.member)
            del parsed
        development_members.append(train9_identity)
        test9 = _expected_member(members, config, split="test", fold=config.data.final_fold)
        holdout_identity = zip_member_identity(test9)
        holdout_identity.update({"fold": config.data.final_fold, "split": "test"})
    if any(is_cf9_test_member(path) for path in opened_members):
        raise AssertionError("target-blind audit opened the CF_9 test payload")
    source_manifest_path = root / "source_manifest.json"
    holdout = {
        "stage": "fi2010_holdout_manifest",
        "created_utc": utc_now(),
        "claim_eligible": bool(source["claim_eligible"] and not config.synthetic),
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "inner_archive_sha256": source["inner_archive_sha256"],
        "config_path": str(config.path),
        "config_sha256": sha256_file(config.path),
        "representation": config.data.representation,
        "member": holdout_identity,
        "payload_opened": False,
    }
    holdout_path = root / "holdout_manifest.json"
    atomic_json(holdout_path, holdout)
    development = {
        "stage": "fi2010_development_manifest",
        "created_utc": utc_now(),
        "claim_eligible": bool(source["claim_eligible"] and not config.synthetic),
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "inner_archive_path": str(inner_path),
        "inner_archive_sha256": source["inner_archive_sha256"],
        "config_path": str(config.path),
        "config_sha256": sha256_file(config.path),
        "representation": config.data.representation,
        "feature_rows": list(config.data.feature_rows),
        "label_rows": list(config.data.label_rows),
        "primary_label_row": config.data.primary_label_row,
        "members": development_members,
        "payload_members_opened": opened_members,
        "cf9_test_payload_opened": False,
        "cumulative_training_policy": "paired_members_only_never_concatenate",
        # Development never opens the holdout manifest; it only carries the audit-time
        # content hash so freeze/release can reject any later metadata redirection.
        "holdout_manifest_sha256": sha256_file(holdout_path),
    }
    development_path = root / "development_manifest.json"
    atomic_json(development_path, development)
    return {
        "development_manifest": str(development_path),
        "development_manifest_sha256": sha256_file(development_path),
        "holdout_manifest": str(holdout_path),
        "holdout_manifest_sha256": sha256_file(holdout_path),
        "development_members": len(development_members),
        "cf9_test_payload_opened": False,
    }


def member_from_identity(
    bundle: zipfile.ZipFile, identity: dict[str, Any]
) -> zipfile.ZipInfo:
    members = validate_zip_directory(bundle)
    path = str(identity["path"])
    if path not in members:
        raise ValueError(f"manifested FI-2010 member is missing: {path}")
    member = members[path]
    if zip_member_identity(member) != {
        key: identity[key]
        for key in ("path", "file_size", "compressed_size", "crc32", "compression")
    }:
        raise ValueError(f"manifested FI-2010 member metadata changed: {path}")
    return member
