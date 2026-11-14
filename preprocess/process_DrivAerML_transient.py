#!/usr/bin/env python3
"""Convert a transient DrivAerML OpenFOAM case into normalized PKL samples.

Geometry is shared by every time step and is stored once in ``geometry.pkl``:

    x_surface:  vehicle boundary-face centres, normalized by surface min/max
    x_volume:   cell centres, normalized by the same surface min/max

Each usable numeric time directory becomes one numbered PKL containing only:

    y_spressure: pressure at the owner cell of each vehicle boundary face
    y_velocity: cell-centred velocity

Pressure and velocity use dataset-wide component-wise mean/std normalization.
The source OpenFOAM case is read-only. Geometry and raw-field caches make a
large mesh resumable without loading the complete case into RAM.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.lib.format import open_memmap

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CASE_DIR = SCRIPT_DIR / "run_3"
DEFAULT_OUTPUT_DIR = DEFAULT_CASE_DIR / "pklFile_transient"
LOGGER = logging.getLogger("process_DrivAerML_transient")
HEADER_BYTES = 128 * 1024


class PreprocessError(RuntimeError):
    """Raised when an OpenFOAM file is missing or structurally inconsistent."""


def read_header(path: Path) -> str:
    """Read enough of an OpenFOAM file to parse its FoamFile/list header."""
    with path.open("rb") as stream:
        return stream.read(HEADER_BYTES).decode("latin-1")


def file_format(path: Path) -> str:
    """Return the OpenFOAM FoamFile format: ``ascii`` or ``binary``."""
    match = re.search(
        r"\bformat\s+(ascii|binary)\s*;",
        read_header(path),
        re.IGNORECASE,
    )
    if match is None:
        raise PreprocessError(f"{path}: FoamFile.format is missing or unsupported")
    return match.group(1).lower()


def header_number(path: Path, pattern: str, description: str) -> int:
    match = re.search(pattern, read_header(path), re.MULTILINE)
    if match is None:
        raise PreprocessError(f"{path}: cannot find {description}")
    return int(match.group(1))


def header_arch(path: Path) -> tuple[str, int, int]:
    """Return endian marker, label bytes and scalar bytes from FoamFile.arch."""
    header = read_header(path)
    arch_match = re.search(r'\barch\s+"([^"]+)"\s*;', header)
    arch = arch_match.group(1) if arch_match else "LSB;label=32;scalar=64"
    endian = "<" if "LSB" in arch.upper() else ">"
    label_match = re.search(r"label\s*=\s*(32|64)", arch)
    scalar_match = re.search(r"scalar\s*=\s*(32|64)", arch)
    label_bytes = int(label_match.group(1)) // 8 if label_match else 4
    scalar_bytes = int(scalar_match.group(1)) // 8 if scalar_match else 8
    return endian, label_bytes, scalar_bytes


def list_data_start(path: Path) -> tuple[int, int]:
    """Return list element count and byte offset immediately after ``(``."""
    with path.open("rb") as stream:
        data = stream.read(HEADER_BYTES)
    text = data.decode("latin-1")
    matches = list(re.finditer(r"(?m)^\s*(\d+)\s*\(", text))
    if not matches:
        raise PreprocessError(f"{path}: cannot locate OpenFOAM list data")
    match = matches[-1]
    return int(match.group(1)), match.end()


def field_list_data_start(path: Path, components: int | None = None) -> tuple[int, int]:
    """Locate the internalField list, never scanning binary payload bytes.

    A generic ``^count (`` search is safe for ASCII mesh lists but can match a
    random newline/number/parenthesis sequence inside a binary vector field.
    Anchoring the search to ``internalField nonuniform List<...>`` avoids that
    ambiguity.
    """
    header = read_header(path)
    match = re.search(
        r"internalField\s+nonuniform\s+List\s*<\s*([A-Za-z0-9_]+)\s*>"
        r"\s*(\d+)\s*\(",
        header,
        re.IGNORECASE,
    )
    if match is None:
        raise PreprocessError(
            f"{path}: cannot locate a nonuniform internalField list"
        )
    field_type = match.group(1).lower()
    expected_type = {1: "scalar", 3: "vector"}.get(components)
    if expected_type is not None and field_type != expected_type:
        raise PreprocessError(
            f"{path}: internalField is List<{field_type}>, expected "
            f"List<{expected_type}>"
        )
    return int(match.group(2)), match.end()


def compact_face_layout(path: Path, label_bytes: int) -> tuple[int, int, int, int]:
    """Locate offsets and vertex labels in a binary faceCompactList."""
    offset_count, offsets_start = list_data_start(path)
    if offset_count < 2:
        raise PreprocessError(
            f"{path}: invalid faceCompactList offset count {offset_count}"
        )
    separator_start = offsets_start + offset_count * label_bytes
    with path.open("rb") as stream:
        stream.seek(separator_start)
        separator = stream.read(128)
    match = re.match(rb"\)\s*(\d+)\s*\(", separator)
    if match is None:
        raise PreprocessError(
            f"{path}: cannot locate vertex-label list in binary faceCompactList"
        )
    label_count = int(match.group(1))
    labels_start = separator_start + match.end()
    return offset_count - 1, offsets_start, labels_start, label_count


def parse_mesh_counts(
    mesh_dir: Path,
) -> tuple[Path, Path, Path, Path, int, int]:
    """Validate polyMesh files and return paths, cell count and face count."""
    poly_mesh = mesh_dir / "constant" / "polyMesh"
    points = poly_mesh / "points"
    faces = poly_mesh / "faces"
    owner = poly_mesh / "owner"
    neighbour = poly_mesh / "neighbour"
    for path in (points, faces, owner, neighbour):
        if not path.is_file():
            raise PreprocessError(f"Mesh file does not exist: {path}")
        print(f"[format] {path.name}: {file_format(path)}")

    owner_count = header_number(
        owner, r"(?m)^\s*(\d+)\s*\(", "owner count"
    )
    cell_match = re.search(r"\bnCells\s*:\s*(\d+)", read_header(owner))
    if cell_match is None:
        raise PreprocessError(f"{owner}: cannot find nCells in FoamFile.note")
    cell_count = int(cell_match.group(1))
    if file_format(faces) == "binary":
        face_count, _, _, _ = compact_face_layout(faces, header_arch(faces)[1])
    else:
        face_count, _ = list_data_start(faces)
    if owner_count != face_count:
        raise PreprocessError(
            f"Mesh owner has {owner_count} entries, but faces has "
            f"{face_count} faces"
        )
    return points, faces, owner, neighbour, cell_count, face_count


def numeric_time_directories(case_dir: Path) -> list[Path]:
    """Return numeric OpenFOAM time directories sorted by numeric value."""
    directories: list[Path] = []
    for path in case_dir.iterdir():
        if not path.is_dir():
            continue
        try:
            float(path.name)
        except ValueError:
            continue
        directories.append(path)
    return sorted(directories, key=lambda item: float(item.name))


def has_cell_list(path: Path) -> bool:
    """Return whether a field contains a nonuniform internal list."""
    try:
        field_list_data_start(path)
    except PreprocessError:
        return False
    return True


@dataclass(frozen=True)
class BoundaryPatch:
    name: str
    patch_type: str
    start_face: int
    num_faces: int

    @property
    def end_face(self) -> int:
        return self.start_face + self.num_faces


@dataclass
class RunningMoments:
    count: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None

    def update(self, values: np.ndarray, chunk_size: int = 1_000_000) -> None:
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError(f"Expected a non-empty (N,C) array, got {values.shape}")
        for start in range(0, len(values), chunk_size):
            batch = np.asarray(values[start : start + chunk_size], dtype=np.float64)
            batch_count = len(batch)
            batch_mean = batch.mean(axis=0)
            centered = batch - batch_mean
            batch_m2 = np.square(centered).sum(axis=0)
            if self.count == 0:
                self.count = batch_count
                self.mean = batch_mean
                self.m2 = batch_m2
                continue
            if self.mean is None or self.m2 is None or self.mean.shape != batch_mean.shape:
                raise RuntimeError("Physical field component count changed")
            delta = batch_mean - self.mean
            total = self.count + batch_count
            self.mean += delta * (batch_count / total)
            self.m2 += batch_m2 + np.square(delta) * self.count * batch_count / total
            self.count = total

    def statistics(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0 or self.mean is None or self.m2 is None:
            raise RuntimeError("No physical values were accumulated")
        std = np.sqrt(np.maximum(self.m2 / self.count, 0.0))
        std[std <= np.finfo(np.float32).eps] = 1.0
        return self.mean.astype(np.float32), std.astype(np.float32)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=None,
        help="Case containing constant/polyMesh; defaults to case-dir",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Geometry/raw-field cache; defaults to output-dir/.cache",
    )
    parser.add_argument(
        "--time",
        nargs="*",
        help="Specific time names. If omitted, use every nonuniform p/U time.",
    )
    parser.add_argument("--pressure", nargs="+", default=["p", "staticP"])
    parser.add_argument("--velocity", nargs="+", default=["U", "velocity"])
    parser.add_argument(
        "--include-patch-regex",
        default=r".*",
        help="Select matching wall patches",
    )
    parser.add_argument(
        "--exclude-patch-regex",
        default=r"^(domain_|CTRL_SURFACE_)",
        help="Exclude non-vehicle wall/control patches",
    )
    parser.add_argument("--face-chunk-size", type=int, default=100_000)
    parser.add_argument("--field-chunk-bytes", type=int, default=64 * 1024**2)
    parser.add_argument("--array-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--rebuild-geometry",
        action="store_true",
        help="Discard compatible geometry cache and recompute centres",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate mesh, fields and patch selection without converting",
    )
    return parser.parse_args()


def choose_field(time_dir: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        candidate = time_dir / name
        if candidate.is_file():
            return candidate
    return None


def parse_boundary_patches(path: Path) -> list[BoundaryPatch]:
    text = path.read_bytes().decode("latin-1")
    pattern = re.compile(
        r"(?m)^\s*([A-Za-z0-9_.:+-]+)\s*\n\s*\{(.*?)^\s*\}",
        re.DOTALL,
    )
    patches: list[BoundaryPatch] = []
    for match in pattern.finditer(text):
        body = match.group(2)
        type_match = re.search(r"\btype\s+([^;\s]+)\s*;", body)
        start_match = re.search(r"\bstartFace\s+(\d+)\s*;", body)
        count_match = re.search(r"\bnFaces\s+(\d+)\s*;", body)
        if not (type_match and start_match and count_match):
            continue
        patches.append(
            BoundaryPatch(
                name=match.group(1),
                patch_type=type_match.group(1),
                start_face=int(start_match.group(1)),
                num_faces=int(count_match.group(1)),
            )
        )
    if not patches:
        raise PreprocessError(f"No boundary patches parsed from {path}")
    return patches


def select_vehicle_patches(
    patches: list[BoundaryPatch], include_pattern: str, exclude_pattern: str
) -> list[BoundaryPatch]:
    include = re.compile(include_pattern)
    exclude = re.compile(exclude_pattern) if exclude_pattern else None
    selected = [
        patch
        for patch in patches
        if patch.patch_type.lower() == "wall"
        and include.search(patch.name)
        and not (exclude and exclude.search(patch.name))
    ]
    if not selected:
        raise PreprocessError(
            "Vehicle patch selection is empty; check include/exclude regexes"
        )
    return selected


def discover_times(
    case_dir: Path,
    requested: list[str] | None,
    pressure_names: list[str],
    velocity_names: list[str],
    cell_count: int,
) -> list[tuple[Path, Path, Path]]:
    time_dirs = numeric_time_directories(case_dir)
    if requested:
        requested_set = set(requested)
        time_dirs = [path for path in time_dirs if path.name in requested_set]
        missing = requested_set - {path.name for path in time_dirs}
        if missing:
            raise PreprocessError(
                f"Requested time directories do not exist: {sorted(missing)}"
            )
    usable: list[tuple[Path, Path, Path]] = []
    for time_dir in time_dirs:
        pressure = choose_field(time_dir, pressure_names)
        velocity = choose_field(time_dir, velocity_names)
        if pressure is None or velocity is None:
            LOGGER.info("Skip time %s: p or U is missing", time_dir.name)
            continue
        if not has_cell_list(pressure) or not has_cell_list(velocity):
            LOGGER.info("Skip time %s: p or U internalField is uniform", time_dir.name)
            continue
        pressure_count, _ = field_list_data_start(pressure, components=1)
        velocity_count, _ = field_list_data_start(velocity, components=3)
        if pressure_count != cell_count or velocity_count != cell_count:
            raise PreprocessError(
                f"{time_dir}: cells={cell_count}, p={pressure_count}, U={velocity_count}"
            )
        usable.append((time_dir, pressure, velocity))
    if not usable:
        raise PreprocessError("No time directory contains nonuniform per-cell p and U")
    return usable


def numpy_dtype(endian: str, byte_width: int, kind: str) -> np.dtype:
    if byte_width not in (4, 8):
        raise PreprocessError(f"Unsupported OpenFOAM width: {byte_width * 8} bits")
    if kind == "float":
        code = "f4" if byte_width == 4 else "f8"
    elif kind == "int":
        code = "i4" if byte_width == 4 else "i8"
    else:
        raise ValueError(kind)
    return np.dtype(("<" if endian == "<" else ">") + code)


def binary_list_memmap(
    path: Path,
    count: int,
    offset: int,
    byte_width: int,
    endian: str,
    components: int = 1,
    kind: str = "int",
) -> np.memmap:
    shape = (count,) if components == 1 else (count, components)
    return np.memmap(
        path,
        mode="r",
        dtype=numpy_dtype(endian, byte_width, kind),
        offset=offset,
        shape=shape,
    )


def open_mesh_arrays(
    points_path: Path,
    faces_path: Path,
    owner_path: Path,
    neighbour_path: Path,
    face_count: int,
):
    for path in (points_path, faces_path, owner_path, neighbour_path):
        if file_format(path) != "binary":
            raise PreprocessError(
                f"Large transient converter requires binary polyMesh; {path} is ASCII"
            )

    point_endian, _, point_bytes = header_arch(points_path)
    point_count, point_start = list_data_start(points_path)
    points = binary_list_memmap(
        points_path,
        point_count,
        point_start,
        point_bytes,
        point_endian,
        components=3,
        kind="float",
    )

    face_endian, face_label_bytes, _ = header_arch(faces_path)
    parsed_face_count, offsets_start, labels_start, label_count = compact_face_layout(
        faces_path, face_label_bytes
    )
    if parsed_face_count != face_count:
        raise PreprocessError("Face count changed while opening compact face list")
    offsets = binary_list_memmap(
        faces_path,
        face_count + 1,
        offsets_start,
        face_label_bytes,
        face_endian,
    )
    labels = binary_list_memmap(
        faces_path,
        label_count,
        labels_start,
        face_label_bytes,
        face_endian,
    )

    owner_endian, owner_bytes, _ = header_arch(owner_path)
    owner_count, owner_start = list_data_start(owner_path)
    if owner_count != face_count:
        raise PreprocessError(f"owner count {owner_count} != face count {face_count}")
    owners = binary_list_memmap(
        owner_path, owner_count, owner_start, owner_bytes, owner_endian
    )

    neighbour_endian, neighbour_bytes, _ = header_arch(neighbour_path)
    neighbour_count, neighbour_start = list_data_start(neighbour_path)
    neighbours = binary_list_memmap(
        neighbour_path,
        neighbour_count,
        neighbour_start,
        neighbour_bytes,
        neighbour_endian,
    )
    return points, offsets, labels, owners, neighbours


def face_centres(
    points: np.ndarray,
    offsets: np.ndarray,
    labels: np.ndarray,
    start_face: int,
    end_face: int,
) -> np.ndarray:
    face_offsets = np.asarray(offsets[start_face : end_face + 1], dtype=np.int64)
    lengths = np.diff(face_offsets)
    if np.any(lengths <= 0):
        raise PreprocessError(f"Empty face found in [{start_face}, {end_face})")
    first_label = int(face_offsets[0])
    last_label = int(face_offsets[-1])
    point_ids = np.asarray(labels[first_label:last_label], dtype=np.int64)
    point_values = np.asarray(points[point_ids], dtype=np.float64)
    local_offsets = face_offsets[:-1] - first_label
    sums = np.add.reduceat(point_values, local_offsets, axis=0)
    return sums / lengths[:, None]


def geometry_signature(
    mesh_paths: tuple[Path, Path, Path, Path],
    cell_count: int,
    face_count: int,
    patches: list[BoundaryPatch],
) -> dict:
    return {
        "cell_count": cell_count,
        "face_count": face_count,
        "mesh_files": {
            path.name: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in mesh_paths
        },
        "patches": [patch.__dict__ for patch in patches],
    }


def build_geometry_cache(
    cache_dir: Path,
    mesh_paths: tuple[Path, Path, Path, Path],
    cell_count: int,
    face_count: int,
    patches: list[BoundaryPatch],
    face_chunk_size: int,
    rebuild: bool,
) -> tuple[Path, Path, Path, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    volume_path = cache_dir / "x_volume_raw.npy"
    surface_path = cache_dir / "x_surface_raw.npy"
    surface_owner_path = cache_dir / "surface_owner.npy"
    metadata_path = cache_dir / "geometry.json"
    signature = geometry_signature(mesh_paths, cell_count, face_count, patches)
    if not rebuild and all(
        path.is_file()
        for path in (volume_path, surface_path, surface_owner_path, metadata_path)
    ):
        with metadata_path.open("r", encoding="utf-8") as stream:
            cached = json.load(stream)
        if cached.get("signature") == signature:
            LOGGER.info("Reuse compatible geometry cache: %s", cache_dir)
            return volume_path, surface_path, surface_owner_path, cached
        LOGGER.warning("Geometry cache signature changed; rebuilding")

    metadata_path.unlink(missing_ok=True)

    points_path, faces_path, owner_path, neighbour_path = mesh_paths
    points, offsets, labels, owners, neighbours = open_mesh_arrays(
        points_path, faces_path, owner_path, neighbour_path, face_count
    )

    sum_path = cache_dir / "cell_face_centre_sum.npy"
    count_path = cache_dir / "cell_face_count.npy"
    cell_sums = open_memmap(sum_path, mode="w+", dtype=np.float64, shape=(cell_count, 3))
    cell_counts = open_memmap(count_path, mode="w+", dtype=np.uint32, shape=(cell_count,))
    started = time.monotonic()
    for start in range(0, face_count, face_chunk_size):
        end = min(start + face_chunk_size, face_count)
        centres = face_centres(points, offsets, labels, start, end)
        owner_ids = np.asarray(owners[start:end], dtype=np.int64)
        np.add.at(cell_sums, owner_ids, centres)
        np.add.at(cell_counts, owner_ids, 1)
        internal_end = min(end, len(neighbours))
        if start < internal_end:
            neighbour_ids = np.asarray(neighbours[start:internal_end], dtype=np.int64)
            internal_centres = centres[: internal_end - start]
            np.add.at(cell_sums, neighbour_ids, internal_centres)
            np.add.at(cell_counts, neighbour_ids, 1)
        if start == 0 or end == face_count or end % 10_000_000 < face_chunk_size:
            LOGGER.info(
                "Geometry faces: %d/%d (%.1f%%), %.1fs",
                end,
                face_count,
                100.0 * end / face_count,
                time.monotonic() - started,
            )
    cell_sums.flush()
    cell_counts.flush()
    for start in range(0, cell_count, max(face_chunk_size, 1_000_000)):
        end = min(start + max(face_chunk_size, 1_000_000), cell_count)
        if np.any(cell_counts[start:end] == 0):
            raise PreprocessError("At least one volume cell has no incident faces")

    x_volume = open_memmap(volume_path, mode="w+", dtype=np.float32, shape=(cell_count, 3))
    array_chunk = max(face_chunk_size, 1_000_000)
    for start in range(0, cell_count, array_chunk):
        end = min(start + array_chunk, cell_count)
        x_volume[start:end] = (
            np.asarray(cell_sums[start:end]) / np.asarray(cell_counts[start:end])[:, None]
        ).astype(np.float32)
    x_volume.flush()

    surface_count = sum(patch.num_faces for patch in patches)
    x_surface = open_memmap(
        surface_path, mode="w+", dtype=np.float32, shape=(surface_count, 3)
    )
    surface_owners = open_memmap(
        surface_owner_path, mode="w+", dtype=np.int64, shape=(surface_count,)
    )
    patch_records = []
    output_start = 0
    for patch in patches:
        patch_output_start = output_start
        for start in range(patch.start_face, patch.end_face, face_chunk_size):
            end = min(start + face_chunk_size, patch.end_face)
            count = end - start
            x_surface[output_start : output_start + count] = face_centres(
                points, offsets, labels, start, end
            ).astype(np.float32)
            surface_owners[output_start : output_start + count] = np.asarray(
                owners[start:end], dtype=np.int64
            )
            output_start += count
        patch_records.append(
            {
                "name": patch.name,
                "start": patch_output_start,
                "count": patch.num_faces,
                "source_start_face": patch.start_face,
            }
        )
        LOGGER.info("Surface patch %-36s faces=%d", patch.name, patch.num_faces)
    x_surface.flush()
    surface_owners.flush()

    metadata = {
        "signature": signature,
        "volume_points": cell_count,
        "surface_points": surface_count,
        "surface_representation": "vehicle_boundary_face_centres",
        "volume_representation": "mean_of_incident_face_centres",
        "surface_pressure_mapping": "boundary_face_owner_cell",
        "patch_records": patch_records,
    }
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    del cell_sums, cell_counts, x_volume, x_surface, surface_owners
    for temporary in (sum_path, count_path):
        temporary.unlink(missing_ok=True)
    return volume_path, surface_path, surface_owner_path, metadata


def read_ascii_internal_field(
    path: Path,
    components: int,
    expected_count: int,
    output: np.memmap,
    chunk_bytes: int,
) -> None:
    count, data_start = field_list_data_start(path, components=components)
    if count != expected_count:
        raise PreprocessError(f"{path}: count={count}, expected={expected_count}")
    translation = bytes.maketrans(b"();", b"   ")
    written = 0
    carry = b""
    closing = re.compile(rb"(?m)^\s*\)\s*;")
    with path.open("rb") as stream:
        stream.seek(data_start)
        finished = False
        while written < expected_count and not finished:
            block = stream.read(chunk_bytes)
            if not block:
                block = b"\n"
                finished = True
            data = carry + block
            last_newline = data.rfind(b"\n")
            if last_newline < 0:
                carry = data
                continue
            complete = data[: last_newline + 1]
            carry = data[last_newline + 1 :]
            marker = closing.search(complete)
            if marker:
                complete = complete[: marker.start()]
                finished = True
            if not complete.strip():
                continue
            clean = complete.translate(translation).decode("ascii")
            values = np.fromstring(clean, dtype=np.float64, sep=" ")
            if values.size == 0:
                continue
            if values.size % components:
                raise PreprocessError(
                    f"{path}: parsed {values.size} values not divisible by {components}"
                )
            rows = values.reshape(-1, components)
            remaining = expected_count - written
            if len(rows) > remaining:
                rows = rows[:remaining]
            output[written : written + len(rows)] = rows.astype(np.float32)
            written += len(rows)
            LOGGER.info("Read %s: %d/%d", path.name, written, expected_count)
    if written != expected_count:
        raise PreprocessError(f"{path}: read {written} rows, expected {expected_count}")
    output.flush()


def read_internal_field_to_npy(
    path: Path,
    components: int,
    expected_count: int,
    output_path: Path,
    chunk_bytes: int,
) -> Path:
    output = open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(expected_count, components),
    )
    if file_format(path) == "ascii":
        read_ascii_internal_field(path, components, expected_count, output, chunk_bytes)
    else:
        count, data_start = field_list_data_start(path, components=components)
        if count != expected_count:
            raise PreprocessError(f"{path}: count={count}, expected={expected_count}")
        endian, _, scalar_bytes = header_arch(path)
        source = binary_list_memmap(
            path,
            count,
            data_start,
            scalar_bytes,
            endian,
            components=components,
            kind="float",
        )
        rows_per_chunk = max(1, chunk_bytes // (scalar_bytes * components))
        for start in range(0, count, rows_per_chunk):
            end = min(start + rows_per_chunk, count)
            values = np.asarray(source[start:end], dtype=np.float32)
            if components == 1:
                values = values[:, None]
            output[start:end] = values
        output.flush()
    del output
    return output_path


def normalize_array(
    source: np.ndarray,
    output_path: Path,
    mean_or_min: np.ndarray,
    std_or_range: np.ndarray,
    chunk_size: int,
) -> Path:
    output = open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=source.shape
    )
    for start in range(0, len(source), chunk_size):
        end = min(start + chunk_size, len(source))
        output[start:end] = (
            (np.asarray(source[start:end], dtype=np.float32) - mean_or_min)
            / std_or_range
        ).astype(np.float32)
    output.flush()
    del output
    return output_path


def atomic_pickle_dump(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def validate_finite_array(
    values: np.ndarray, name: str, path: Path, chunk_size: int
) -> None:
    for start in range(0, len(values), chunk_size):
        if not np.isfinite(values[start : start + chunk_size]).all():
            raise RuntimeError(f"{path}: {name} contains NaN/Inf")


def validate_geometry(geometry: dict, path: Path, chunk_size: int) -> None:
    required = {"x_surface", "x_volume"}
    missing = required - set(geometry)
    if missing:
        raise RuntimeError(f"{path}: missing geometry fields {sorted(missing)}")
    for name in required:
        values = geometry[name]
        if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
            raise RuntimeError(f"{path}: invalid {name} shape {values.shape}")
        validate_finite_array(values, name, path, chunk_size)


def validate_time_sample(
    sample: dict,
    path: Path,
    surface_points: int,
    volume_points: int,
    chunk_size: int,
) -> None:
    required = {"y_spressure", "y_velocity"}
    missing = required - set(sample)
    if missing:
        raise RuntimeError(f"{path}: missing physical fields {sorted(missing)}")
    expected_shapes = {
        "y_spressure": (surface_points, 1),
        "y_velocity": (volume_points, 3),
    }
    for name in required:
        values = sample[name]
        if tuple(values.shape) != expected_shapes[name]:
            raise RuntimeError(
                f"{path}: {name} shape {values.shape} != {expected_shapes[name]}"
            )
        validate_finite_array(values, name, path, chunk_size)


def field_dimensions(path: Path) -> str | None:
    match = re.search(r"\bdimensions\s+(\[[^;]+\])\s*;", read_header(path))
    return match.group(1) if match else None


def process(args: argparse.Namespace) -> None:
    case_dir = args.case_dir.expanduser().resolve()
    mesh_dir = (args.mesh_dir or case_dir).expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    cache_dir = (
        args.cache_dir.expanduser().resolve()
        if args.cache_dir is not None
        else case_dir / ".process_DrivAerML_transient_cache"
    )
    if args.face_chunk_size <= 0 or args.array_chunk_size <= 0:
        raise ValueError("Chunk sizes must be positive")

    mesh_paths = parse_mesh_counts(mesh_dir)
    points, faces, owner, neighbour, cell_count, face_count = mesh_paths
    mesh_files = (points, faces, owner, neighbour)
    boundary_path = mesh_dir / "constant" / "polyMesh" / "boundary"
    patches = parse_boundary_patches(boundary_path)
    selected = select_vehicle_patches(
        patches, args.include_patch_regex, args.exclude_patch_regex
    )
    times = discover_times(
        case_dir,
        args.time,
        args.pressure,
        args.velocity,
        cell_count,
    )

    print("=" * 80)
    print("Transient DrivAerML preprocessing")
    print(f"Case              : {case_dir}")
    print(f"Output            : {output_dir}")
    print(f"Cache             : {cache_dir}")
    print(f"Cells/faces       : {cell_count:,} / {face_count:,}")
    print(f"Selected patches  : {len(selected)} ({sum(p.num_faces for p in selected):,} faces)")
    print(f"Times             : {[item[0].name for item in times]}")
    print("Coordinates       : geometry.pkl, shared vehicle-surface min/max")
    print("Time fields       : one numbered PKL per time, without coordinates")
    print("Physical fields   : global component-wise mean/std")
    print("=" * 80)
    for patch in selected:
        print(f"  {patch.name:<40} {patch.num_faces:>10,} faces")
    if args.check_only:
        print("Check completed; no files were written.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("[0-9]*.pkl"))
    geometry_path = output_dir / "geometry.pkl"
    scalar_path = output_dir / "normalization_scalars.pkl"
    if (existing or geometry_path.exists() or scalar_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already contains outputs; use --overwrite"
        )
    if args.overwrite:
        for path in existing:
            path.unlink()
        geometry_path.unlink(missing_ok=True)
        scalar_path.unlink(missing_ok=True)
        for path in output_dir.glob("*.pkl.tmp"):
            path.unlink()

    volume_path, surface_path, surface_owner_path, geometry = build_geometry_cache(
        cache_dir,
        mesh_files,
        cell_count,
        face_count,
        selected,
        args.face_chunk_size,
        args.rebuild_geometry,
    )
    x_volume_raw = np.load(volume_path, mmap_mode="r")
    x_surface_raw = np.load(surface_path, mmap_mode="r")
    surface_owner = np.load(surface_owner_path, mmap_mode="r")
    surface_min = np.asarray(x_surface_raw).min(axis=0).astype(np.float32)
    surface_max = np.asarray(x_surface_raw).max(axis=0).astype(np.float32)
    surface_range = surface_max - surface_min
    surface_range[surface_range <= np.finfo(np.float32).eps] = 1.0
    volume_physical_min = np.asarray(x_volume_raw).min(axis=0).astype(np.float32)
    volume_physical_max = np.asarray(x_volume_raw).max(axis=0).astype(np.float32)

    x_surface_normalized_path = normalize_array(
        x_surface_raw,
        cache_dir / "x_surface_normalized.npy",
        surface_min,
        surface_range,
        args.array_chunk_size,
    )
    x_volume_normalized_path = normalize_array(
        x_volume_raw,
        cache_dir / "x_volume_normalized.npy",
        surface_min,
        surface_range,
        args.array_chunk_size,
    )

    pressure_moments = RunningMoments()
    velocity_moments = RunningMoments()
    records: list[dict] = []
    raw_fields: list[tuple[Path, Path]] = []
    for sample_id, (time_dir, pressure, velocity) in enumerate(times, start=1):
        started = time.monotonic()
        pressure_internal_path = cache_dir / f"time_{sample_id}_pressure_internal.npy"
        velocity_path = cache_dir / f"time_{sample_id}_velocity.npy"
        surface_pressure_path = cache_dir / f"time_{sample_id}_surface_pressure.npy"
        read_internal_field_to_npy(
            pressure,
            1,
            cell_count,
            pressure_internal_path,
            args.field_chunk_bytes,
        )
        read_internal_field_to_npy(
            velocity,
            3,
            cell_count,
            velocity_path,
            args.field_chunk_bytes,
        )
        pressure_internal = np.load(pressure_internal_path, mmap_mode="r")
        surface_pressure = open_memmap(
            surface_pressure_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(surface_owner), 1),
        )
        for start in range(0, len(surface_owner), args.array_chunk_size):
            end = min(start + args.array_chunk_size, len(surface_owner))
            ids = np.asarray(surface_owner[start:end], dtype=np.int64)
            surface_pressure[start:end] = pressure_internal[ids]
        surface_pressure.flush()
        del pressure_internal, surface_pressure
        pressure_internal_path.unlink(missing_ok=True)

        surface_pressure = np.load(surface_pressure_path, mmap_mode="r")
        velocity_values = np.load(velocity_path, mmap_mode="r")
        pressure_moments.update(surface_pressure, args.array_chunk_size)
        velocity_moments.update(velocity_values, args.array_chunk_size)
        raw_fields.append((surface_pressure_path, velocity_path))
        records.append(
            {
                "sample_id": sample_id,
                "output_file": f"{sample_id}.pkl",
                "time_name": time_dir.name,
                "time_value": float(time_dir.name),
                "pressure_file": pressure.name,
                "velocity_file": velocity.name,
                "surface_points": len(surface_pressure),
                "volume_points": len(velocity_values),
                "pressure_dimensions": field_dimensions(pressure),
                "velocity_dimensions": field_dimensions(velocity),
            }
        )
        LOGGER.info(
            "Fields [%d/%d] time=%s completed in %.1fs",
            sample_id,
            len(times),
            time_dir.name,
            time.monotonic() - started,
        )

    pressure_mean, pressure_std = pressure_moments.statistics()
    velocity_mean, velocity_std = velocity_moments.statistics()
    scalars = {
        "schema_version": "drivaerml-transient-split-geometry-v2",
        "geometry_file": geometry_path.name,
        "coordinate_scale_policy": "surface_scale_for_surface_and_volume",
        "coords_min": surface_min,
        "coords_max": surface_max,
        "coords_range": surface_range,
        "surface_coords_min": surface_min,
        "surface_coords_max": surface_max,
        "surface_coords_range": surface_range,
        "volume_coords_min": surface_min,
        "volume_coords_max": surface_max,
        "volume_coords_range": surface_range,
        "volume_physical_coords_min": volume_physical_min,
        "volume_physical_coords_max": volume_physical_max,
        "y_spressure_mean": pressure_mean,
        "y_spressure_std": pressure_std,
        "y_velocity_mean": velocity_mean,
        "y_velocity_std": velocity_std,
        "samples": records,
        "surface_patches": geometry["patch_records"],
        "surface_representation": geometry["surface_representation"],
        "volume_representation": geometry["volume_representation"],
        "surface_pressure_mapping": geometry["surface_pressure_mapping"],
        "normalization": {
            "x_surface": "global_vehicle_surface_minmax",
            "x_volume": "global_vehicle_surface_minmax",
            "y_spressure": "global_mean_std",
            "y_velocity": "global_mean_std_per_component",
        },
    }

    x_surface_normalized = np.load(x_surface_normalized_path, mmap_mode="r")
    x_volume_normalized = np.load(x_volume_normalized_path, mmap_mode="r")
    geometry_sample = {
        "schema_version": scalars["schema_version"],
        "x_surface": np.asarray(x_surface_normalized),
        "x_volume": np.asarray(x_volume_normalized),
        "surface_patches": geometry["patch_records"],
        "surface_representation": geometry["surface_representation"],
        "volume_representation": geometry["volume_representation"],
        "coordinate_scale_policy": scalars["coordinate_scale_policy"],
        "normalization": {
            "x_surface": scalars["normalization"]["x_surface"],
            "x_volume": scalars["normalization"]["x_volume"],
        },
    }
    validate_geometry(geometry_sample, geometry_path, args.array_chunk_size)
    atomic_pickle_dump(geometry_sample, geometry_path)
    LOGGER.info(
        "Write %s: surface=%d, volume=%d",
        geometry_path.name,
        len(geometry_sample["x_surface"]),
        len(geometry_sample["x_volume"]),
    )
    del geometry_sample

    for record, (surface_pressure_path, velocity_path) in zip(records, raw_fields):
        sample_id = record["sample_id"]
        pressure_raw = np.load(surface_pressure_path, mmap_mode="r")
        velocity_raw = np.load(velocity_path, mmap_mode="r")
        pressure_normalized_path = normalize_array(
            pressure_raw,
            cache_dir / f"time_{sample_id}_surface_pressure_normalized.npy",
            pressure_mean,
            pressure_std,
            args.array_chunk_size,
        )
        velocity_normalized_path = normalize_array(
            velocity_raw,
            cache_dir / f"time_{sample_id}_velocity_normalized.npy",
            velocity_mean,
            velocity_std,
            args.array_chunk_size,
        )
        pressure_normalized = np.load(pressure_normalized_path, mmap_mode="r")
        velocity_normalized = np.load(velocity_normalized_path, mmap_mode="r")
        sample = {
            "schema_version": scalars["schema_version"],
            "sample_id": sample_id,
            "time_name": record["time_name"],
            "time_value": record["time_value"],
            "y_spressure": np.asarray(pressure_normalized),
            "y_velocity": np.asarray(velocity_normalized),
            "geometry_file": geometry_path.name,
            "normalization_scalars_file": scalar_path.name,
            "source": {
                "case_dir": str(case_dir),
                "time_directory": record["time_name"],
                "pressure_file": record["pressure_file"],
                "velocity_file": record["velocity_file"],
            },
            "normalization": {
                "y_spressure": scalars["normalization"]["y_spressure"],
                "y_velocity": scalars["normalization"]["y_velocity"],
            },
        }
        destination = output_dir / f"{sample_id}.pkl"
        validate_time_sample(
            sample,
            destination,
            surface_points=len(x_surface_normalized),
            volume_points=len(x_volume_normalized),
            chunk_size=args.array_chunk_size,
        )
        atomic_pickle_dump(sample, destination)
        LOGGER.info(
            "Write %s: time=%s, pressure=%d, velocity=%d",
            destination.name,
            record["time_name"],
            len(sample["y_spressure"]),
            len(sample["y_velocity"]),
        )
        del sample, pressure_normalized, velocity_normalized
        Path(pressure_normalized_path).unlink(missing_ok=True)
        Path(velocity_normalized_path).unlink(missing_ok=True)

    atomic_pickle_dump(scalars, scalar_path)
    print("Conversion complete.")
    print(f"Geometry             : {geometry_path}")
    print(f"Time samples         : {len(records)}")
    print(f"Output               : {output_dir}")
    print(f"Normalization scalars: {scalar_path}")
    print("Source OpenFOAM files were preserved.")


def main() -> int:
    args = parse_args()
    try:
        process(args)
        return 0
    except (OSError, ValueError, RuntimeError, PreprocessError) as error:
        LOGGER.error("%s", error)
        return 1


if __name__ == "__main__":
    configure_logging()
    raise SystemExit(main())
