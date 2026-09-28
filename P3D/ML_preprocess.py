"""Convert OpenFOAM volume fields into aligned point-cloud text files.

The default OpenFOAM fields are cell-centred: ``p`` contains one value per
cell and ``U`` contains one vector per cell.  This script therefore computes
one centre for every mesh cell from ``constant/polyMesh`` and writes rows in
the same order as the field values:

    <time>_position   Nx3, x y z
    <time>_pressure   Nx1, p
    <time>_velocity   Nx3, Ux Uy Uz

The files are whitespace-separated text without a header.  The time part of
each filename is copied from the corresponding OpenFOAM time-directory name.
Mixed-format cases are supported: the mesh may be binary while the result
fields are ASCII, or vice versa.  For binary files, the ``arch`` entry in the
OpenFOAM header is used to determine byte order and the sizes of ``label``
and ``scalar``.

Example::

    python ML_preprocess.py \
        --input-dir /work/mae-zhangbj/OpenFOAM/mae-zhangbj-v2212/run/car/DrivAerML_car/run_2 \
        --output-dir /work/mae-zhangbj/OpenFOAM/mae-zhangbj-v2212/run/car/DrivAerML_car/run_2/point_cloud

By default, the output directory is ``<input-dir>/point_cloud`` and all
numeric time directories are processed.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np


FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
FLOAT_RE = re.compile(FLOAT_PATTERN)
VECTOR_RE = re.compile(
    rf"\(\s*({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\s*\)"
)
FORMAT_BINARY_RE = re.compile(rb"\bformat\s+binary\b")
ARCH_RE = re.compile(rb'\barch\s+"([^"]+)"')


class OpenFOAMFormatError(ValueError):
    """Raised when an OpenFOAM file does not have the expected ASCII format."""


def strip_comments(text: str) -> str:
    """Remove OpenFOAM line and block comments."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//.*", "", text)


def read_bytes(path: Path) -> bytes:
    """Read an OpenFOAM file as bytes."""
    try:
        return path.read_bytes()
    except OSError as error:
        raise OpenFOAMFormatError(
            f"Could not read OpenFOAM file {path}: {error}"
        ) from error


def is_binary(raw: bytes) -> bool:
    """Return whether the OpenFOAM header declares binary format."""
    return FORMAT_BINARY_RE.search(raw[:4096]) is not None


def read_ascii(path: Path) -> str:
    """Read a text OpenFOAM file."""
    raw = read_bytes(path)
    if is_binary(raw):
        raise OpenFOAMFormatError(f"{path} is a binary OpenFOAM file")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise OpenFOAMFormatError(f"{path} is not valid UTF-8 OpenFOAM text") from error
    return strip_comments(text)


def binary_settings(raw: bytes) -> tuple[str, np.dtype, np.dtype]:
    """Read endian, label dtype, and scalar dtype from a binary header."""
    arch_match = ARCH_RE.search(raw[:4096])
    arch = arch_match.group(1).decode("ascii", errors="replace") if arch_match else ""
    endian = ">" if "MSB" in arch else "<"

    label_match = re.search(r"label\s*=\s*(32|64)", arch)
    scalar_match = re.search(r"scalar\s*=\s*(32|64)", arch)
    label_bits = int(label_match.group(1)) if label_match else 32
    scalar_bits = int(scalar_match.group(1)) if scalar_match else 64
    return (
        endian,
        np.dtype(f"{endian}i{label_bits // 8}"),
        np.dtype(f"{endian}f{scalar_bits // 8}"),
    )


def binary_list_start(raw: bytes, keyword: str) -> tuple[int, int, np.dtype, np.dtype]:
    """Find a binary OpenFOAM list count and its payload offset."""
    pattern = re.compile(
        rb"\b" + re.escape(keyword.encode("ascii")) + rb"\b" rb".*?\b(\d+)\s*\(",
        flags=re.DOTALL,
    )
    match = pattern.search(raw[:4096])
    if match is None:
        raise OpenFOAMFormatError(f"Could not find binary list '{keyword}'")

    count = int(match.group(1))
    offset = match.end()
    while offset < len(raw) and raw[offset] in b" \t\r\n":
        offset += 1
    endian, label_dtype, scalar_dtype = binary_settings(raw)
    return count, offset, label_dtype, scalar_dtype


def binary_values(
    raw: bytes, offset: int, count: int, dtype: np.dtype, path: Path
) -> np.ndarray:
    """Read ``count`` primitive values from a binary OpenFOAM list."""
    byte_count = count * dtype.itemsize
    end = offset + byte_count
    if end > len(raw):
        raise OpenFOAMFormatError(
            f"{path}: binary list is shorter than its declared size {count}"
        )
    return np.frombuffer(raw, dtype=dtype, count=count, offset=offset).copy()


def parse_binary_faces_at(
    raw: bytes, count: int, offset: int, label_dtype: np.dtype, path: Path
) -> list[np.ndarray] | None:
    """Try the usual variable-length binary face-list representation."""
    faces: list[np.ndarray] = []
    try:
        for _ in range(count):
            face_size = int(binary_values(raw, offset, 1, label_dtype, path)[0])
            if face_size <= 0 or face_size > 100000:
                return None
            offset += label_dtype.itemsize
            face = binary_values(raw, offset, face_size, label_dtype, path)
            if np.any(face < 0):
                return None
            offset += face_size * label_dtype.itemsize
            faces.append(face.astype(np.int64, copy=False))
    except (OpenFOAMFormatError, ValueError, OverflowError):
        return None
    return faces


def parse_compact_binary_faces(
    raw: bytes, count: int, offset: int, label_dtype: np.dtype, path: Path
) -> list[np.ndarray] | None:
    """Try compact layouts with sizes or offsets followed by vertex labels."""
    try:
        sizes = binary_values(raw, offset, count, label_dtype, path).astype(
            np.int64, copy=False
        )
    except OpenFOAMFormatError:
        return None
    if np.all(sizes > 0) and np.all(sizes <= 100000):
        labels_offset = offset + count * label_dtype.itemsize
        total_labels = int(sizes.sum())
    else:
        try:
            offsets = binary_values(raw, offset, count + 1, label_dtype, path).astype(
                np.int64, copy=False
            )
        except OpenFOAMFormatError:
            return None
        if (
            offsets[0] != 0
            or np.any(np.diff(offsets) <= 0)
            or np.any(np.diff(offsets) > 100000)
        ):
            return None
        sizes = np.diff(offsets)
        labels_offset = offset + (count + 1) * label_dtype.itemsize
        total_labels = int(offsets[-1])

    try:
        labels = binary_values(raw, labels_offset, total_labels, label_dtype, path)
    except OpenFOAMFormatError:
        return None
    if np.any(labels < 0):
        return None

    faces: list[np.ndarray] = []
    label_start = 0
    for face_size in sizes:
        label_end = label_start + int(face_size)
        faces.append(labels[label_start:label_end].astype(np.int64, copy=False))
        label_start = label_end
    return faces


def parse_binary_faces(path: Path) -> list[np.ndarray]:
    """Parse regular or compact binary OpenFOAM face lists."""
    raw = read_bytes(path)
    count, offset, label_dtype, _ = binary_list_start(raw, "faces")
    candidates = [offset]
    for extra in range(1, 17):
        candidates.append(offset + extra)

    for candidate in candidates:
        faces = parse_binary_faces_at(raw, count, candidate, label_dtype, path)
        if faces is not None:
            return faces
        faces = parse_compact_binary_faces(raw, count, candidate, label_dtype, path)
        if faces is not None:
            return faces

    raise OpenFOAMFormatError(
        f"{path}: unsupported binary face-list layout; the file is valid OpenFOAM "
        "data, but its compact encoding could not be decoded"
    )


def matching_parentheses(text: str, opening_index: int) -> str:
    """Return the text enclosed by the parenthesis at ``opening_index``."""
    if opening_index >= len(text) or text[opening_index] != "(":
        raise OpenFOAMFormatError("Expected an opening parenthesis in OpenFOAM data")

    depth = 0
    for index in range(opening_index, len(text)):
        character = text[index]
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return text[opening_index + 1 : index]
    raise OpenFOAMFormatError("Unclosed parenthesis in OpenFOAM data")


def list_after_keyword(text: str, keyword: str) -> tuple[int, str]:
    """Find ``count ( ... )`` after a named mesh object."""
    keyword_match = re.search(rf"\b{re.escape(keyword)}\b", text)
    if keyword_match is None:
        raise OpenFOAMFormatError(f"Could not find '{keyword}' in mesh file")

    list_match = re.search(r"\b(\d+)\s*\(", text[keyword_match.end() :])
    if list_match is None:
        raise OpenFOAMFormatError(f"Could not find the list for '{keyword}'")

    count = int(list_match.group(1))
    opening_index = keyword_match.end() + list_match.end() - 1
    return count, matching_parentheses(text, opening_index)


def parse_points(path: Path) -> np.ndarray:
    """Parse ``constant/polyMesh/points`` into an ``(N, 3)`` array."""
    raw = read_bytes(path)
    if is_binary(raw):
        count, offset, _, scalar_dtype = binary_list_start(raw, "points")
        values = binary_values(raw, offset, count * 3, scalar_dtype, path)
        return values.reshape(count, 3).astype(np.float64, copy=False)

    count, body = list_after_keyword(read_ascii(path), "points")
    matches = VECTOR_RE.findall(body)
    points = np.asarray(matches, dtype=np.float64)
    if points.shape != (count, 3):
        raise OpenFOAMFormatError(
            f"{path}: header declares {count} points, parsed {len(points)}"
        )
    return points


def parse_owner(path: Path) -> np.ndarray:
    """Parse cell owner indices from ``owner``."""
    raw = read_bytes(path)
    if is_binary(raw):
        count, offset, label_dtype, _ = binary_list_start(raw, "owner")
        return binary_values(raw, offset, count, label_dtype, path).astype(
            np.int64, copy=False
        )

    count, body = list_after_keyword(read_ascii(path), "owner")
    values = np.fromiter(
        (int(value) for value in re.findall(r"[+-]?\d+", body)),
        dtype=np.int64,
    )
    if values.size != count:
        raise OpenFOAMFormatError(
            f"{path}: header declares {count} owner entries, parsed {values.size}"
        )
    return values


def parse_faces(path: Path) -> list[np.ndarray]:
    """Parse the vertex indices of every face from ``faces``."""
    raw = read_bytes(path)
    if is_binary(raw):
        return parse_binary_faces(path)

    count, body = list_after_keyword(read_ascii(path), "faces")
    faces: list[np.ndarray] = []
    for face_text in re.findall(r"(?:\d+\s*)?\(([^()]*)\)", body):
        values = np.fromiter(
            (int(value) for value in re.findall(r"[+-]?\d+", face_text)),
            dtype=np.int64,
        )
        if values.size == 0:
            raise OpenFOAMFormatError(f"{path}: found an empty face")
        faces.append(values)

    if len(faces) != count:
        raise OpenFOAMFormatError(
            f"{path}: header declares {count} faces, parsed {len(faces)}"
        )
    return faces


def parse_neighbour(path: Path) -> np.ndarray:
    """Parse neighbour indices, allowing an empty boundary-only list."""
    raw = read_bytes(path)
    if is_binary(raw):
        count, offset, label_dtype, _ = binary_list_start(raw, "neighbour")
        return binary_values(raw, offset, count, label_dtype, path).astype(
            np.int64, copy=False
        )

    count, body = list_after_keyword(read_ascii(path), "neighbour")
    values = np.fromiter(
        (int(value) for value in re.findall(r"[+-]?\d+", body)),
        dtype=np.int64,
    )
    if values.size != count:
        raise OpenFOAMFormatError(
            f"{path}: header declares {count} neighbour entries, parsed {values.size}"
        )
    return values


def cell_centres(
    points: np.ndarray,
    faces: list[np.ndarray],
    owner: np.ndarray,
    neighbour: np.ndarray,
) -> np.ndarray:
    """Calculate one geometric cell centre per OpenFOAM cell.

    The centre is the arithmetic mean of the unique vertices belonging to a
    cell.  This is stable for arbitrary polyhedral cells and is sufficient for
    aligning the cell-centred ``p`` and ``U`` fields with spatial coordinates.
    """
    if len(faces) != owner.size:
        raise OpenFOAMFormatError("The number of faces and owner entries differ")
    if neighbour.size > owner.size:
        raise OpenFOAMFormatError("The neighbour list cannot exceed the face list")

    cell_count = int(owner.max()) + 1 if owner.size else 0
    if neighbour.size:
        cell_count = max(cell_count, int(neighbour.max()) + 1)

    cell_vertices: list[set[int]] = [set() for _ in range(cell_count)]
    for face_index, face_vertices in enumerate(faces):
        owner_cell = int(owner[face_index])
        if owner_cell < 0 or owner_cell >= cell_count:
            raise OpenFOAMFormatError(
                f"Face {face_index} references an invalid owner cell {owner_cell}"
            )
        cell_vertices[owner_cell].update(int(value) for value in face_vertices)
        if face_index < neighbour.size:
            neighbour_cell = int(neighbour[face_index])
            if neighbour_cell < 0 or neighbour_cell >= cell_count:
                raise OpenFOAMFormatError(
                    f"Face {face_index} references an invalid neighbour cell "
                    f"{neighbour_cell}"
                )
            cell_vertices[neighbour_cell].update(int(value) for value in face_vertices)

    centres = np.empty((cell_count, 3), dtype=np.float64)
    for cell_index, vertex_indices in enumerate(cell_vertices):
        if not vertex_indices:
            raise OpenFOAMFormatError(
                f"Cell {cell_index} has no vertices; mesh topology is invalid"
            )
        indices = np.fromiter(vertex_indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= len(points)):
            raise OpenFOAMFormatError(f"Cell {cell_index} references an invalid point")
        centres[cell_index] = points[indices].mean(axis=0)
    return centres


def parse_field(path: Path) -> tuple[np.ndarray, bool]:
    """Parse an OpenFOAM scalar/vector ``internalField``.

    Returns ``(values, is_vector)``.  Uniform fields return one value and are
    expanded after the mesh size is known.
    """
    raw = read_bytes(path)
    if is_binary(raw):
        internal_field_match = re.search(rb"\binternalField\b", raw[:4096])
        if internal_field_match is None:
            raise OpenFOAMFormatError(f"{path}: missing internalField")
        header_tail = raw[internal_field_match.end() : min(len(raw), 4096)]
        uniform_match = re.match(rb"\s*uniform\s+", header_tail)
        if uniform_match:
            value_text = header_tail[uniform_match.end() :].lstrip()
            vector_match = VECTOR_RE.match(value_text.decode("ascii", errors="ignore"))
            if vector_match:
                values = np.asarray(vector_match.groups(), dtype=np.float64).reshape(
                    1, 3
                )
                return values, True
            scalar_match = FLOAT_RE.match(value_text.decode("ascii", errors="ignore"))
            if scalar_match:
                return (
                    np.asarray([[float(scalar_match.group())]], dtype=np.float64),
                    False,
                )
            raise OpenFOAMFormatError(f"{path}: could not parse uniform internalField")

        list_match = re.search(rb"\bList\s*<\s*(scalar|vector)\s*>", header_tail)
        if list_match is None:
            raise OpenFOAMFormatError(f"{path}: unsupported binary internalField type")
        count, offset, _, scalar_dtype = binary_list_start(raw, "internalField")
        is_vector = list_match.group(1) == b"vector"
        component_count = 3 if is_vector else 1
        values = binary_values(raw, offset, count * component_count, scalar_dtype, path)
        return (
            values.reshape(count, component_count).astype(np.float64, copy=False),
            is_vector,
        )

    text = read_ascii(path)
    field_match = re.search(r"\binternalField\b", text)
    if field_match is None:
        raise OpenFOAMFormatError(f"{path}: missing internalField")

    remainder = text[field_match.end() :]
    uniform_match = re.match(r"\s*uniform\s+", remainder)
    if uniform_match:
        value_text = remainder[uniform_match.end() :]
        vector_match = VECTOR_RE.match(value_text.strip())
        if vector_match:
            values = np.asarray(vector_match.groups(), dtype=np.float64).reshape(1, 3)
            return values, True
        scalar_match = FLOAT_RE.match(value_text.strip())
        if scalar_match:
            return np.asarray([[float(scalar_match.group())]], dtype=np.float64), False
        raise OpenFOAMFormatError(f"{path}: could not parse uniform internalField")

    nonuniform_match = re.match(r"\s*nonuniform\s+List<[^>]+>\s+(\d+)\s*\(", remainder)
    if nonuniform_match is None:
        raise OpenFOAMFormatError(
            f"{path}: internalField must be ASCII uniform or nonuniform List"
        )

    expected_count = int(nonuniform_match.group(1))
    opening_index = nonuniform_match.end() - 1
    body = matching_parentheses(remainder, opening_index)
    vectors = VECTOR_RE.findall(body)
    if vectors:
        values = np.asarray(vectors, dtype=np.float64)
        is_vector = True
    else:
        scalars = FLOAT_RE.findall(body)
        values = np.asarray(scalars, dtype=np.float64).reshape(-1, 1)
        is_vector = False

    if values.shape[0] != expected_count:
        raise OpenFOAMFormatError(
            f"{path}: header declares {expected_count} field values, parsed "
            f"{values.shape[0]}"
        )
    return values, is_vector


def expand_field(values: np.ndarray, expected_count: int, path: Path) -> np.ndarray:
    """Expand a uniform value or validate a nonuniform field length."""
    if values.shape[0] == 1:
        return np.repeat(values, expected_count, axis=0)
    if values.shape[0] != expected_count:
        raise OpenFOAMFormatError(
            f"{path}: field has {values.shape[0]} values, but mesh has "
            f"{expected_count} cells"
        )
    return values


def numeric_time_directories(input_dir: Path) -> list[Path]:
    """Return OpenFOAM time directories sorted by numeric time."""
    directories: list[tuple[float, Path]] = []
    for path in input_dir.iterdir():
        if not path.is_dir():
            continue
        try:
            time_value = float(path.name)
        except ValueError:
            continue
        directories.append((time_value, path))
    return [path for _, path in sorted(directories, key=lambda item: item[0])]


def save_array(path: Path, values: np.ndarray) -> None:
    """Save one point-cloud component as whitespace-separated text."""
    np.savetxt(path, values, fmt="%.10e")


def convert(
    input_dir: Path,
    output_dir: Path,
    pressure_name: str,
    velocity_name: str,
    time_directories: Iterable[Path],
) -> int:
    """Convert selected OpenFOAM time directories and return their count."""
    mesh_dir = input_dir / "constant" / "polyMesh"
    required_mesh_files = (mesh_dir / "points", mesh_dir / "faces", mesh_dir / "owner")
    missing_mesh_files = [path for path in required_mesh_files if not path.is_file()]
    if missing_mesh_files:
        missing = ", ".join(str(path) for path in missing_mesh_files)
        raise FileNotFoundError(f"Missing required mesh file(s): {missing}")

    points = parse_points(mesh_dir / "points")
    faces = parse_faces(mesh_dir / "faces")
    owner = parse_owner(mesh_dir / "owner")
    neighbour_path = mesh_dir / "neighbour"
    neighbour = (
        parse_neighbour(neighbour_path)
        if neighbour_path.is_file()
        else np.empty(0, dtype=np.int64)
    )
    centres = cell_centres(points, faces, owner, neighbour)

    output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    for time_dir in time_directories:
        pressure_path = time_dir / pressure_name
        velocity_path = time_dir / velocity_name
        if not pressure_path.is_file() or not velocity_path.is_file():
            print(
                f"skip {time_dir.name}: missing {pressure_name} or {velocity_name}",
            )
            continue

        pressure, pressure_is_vector = parse_field(pressure_path)
        velocity, velocity_is_vector = parse_field(velocity_path)
        if pressure_is_vector:
            raise OpenFOAMFormatError(f"{pressure_path}: pressure must be scalar")
        if not velocity_is_vector or velocity.shape[1] != 3:
            raise OpenFOAMFormatError(f"{velocity_path}: velocity must be a 3D vector")

        pressure = expand_field(pressure, len(centres), pressure_path)
        velocity = expand_field(velocity, len(centres), velocity_path)
        save_array(output_dir / f"{time_dir.name}_position", centres)
        save_array(output_dir / f"{time_dir.name}_pressure", pressure)
        save_array(output_dir / f"{time_dir.name}_velocity", velocity)
        processed += 1
        print(f"processed {time_dir.name}: {len(centres)} points")

    if processed == 0:
        raise FileNotFoundError(
            f"No time directory containing '{pressure_name}' and '{velocity_name}' was found"
        )
    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert OpenFOAM p/U cell fields to aligned point-cloud files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="OpenFOAM case directory containing constant/polyMesh and time folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to <input-dir>/point_cloud.",
    )
    parser.add_argument(
        "--pressure-name",
        default="p",
        help="OpenFOAM pressure field filename (default: p).",
    )
    parser.add_argument(
        "--velocity-name",
        default="U",
        help="OpenFOAM velocity field filename (default: U).",
    )
    parser.add_argument(
        "--time",
        dest="times",
        action="append",
        help="Process only this time directory; may be supplied multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input case directory does not exist: {input_dir}")

    output_dir = args.output_dir or input_dir / "point_cloud"
    output_dir = output_dir.expanduser().resolve()
    all_time_directories = numeric_time_directories(input_dir)
    if args.times:
        selected_names = set(args.times)
        time_directories = [
            path for path in all_time_directories if path.name in selected_names
        ]
        missing_names = selected_names - {path.name for path in time_directories}
        if missing_names:
            missing = ", ".join(sorted(missing_names))
            raise FileNotFoundError(f"Requested time directory not found: {missing}")
    else:
        time_directories = all_time_directories

    count = convert(
        input_dir=input_dir,
        output_dir=output_dir,
        pressure_name=args.pressure_name,
        velocity_name=args.velocity_name,
        time_directories=time_directories,
    )
    print(f"done: {count} time step(s), output directory: {output_dir}")


if __name__ == "__main__":
    main()
