#!/usr/bin/env python3
"""Download a selected DrivAerStar case range from Hugging Face.

The current Hugging Face repository stores each case as five separate files,
not as ``case_E_batch_*.zip`` archives. This script discovers the case IDs
that actually exist, downloads only the requested range, and can optionally
create one local ZIP archive per case.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, get_token, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from requests import RequestException


REPO_ID = "drivaerstar000/drivaerstar"
REPO_TYPE = "dataset"
DEFAULT_ENDPOINT = "https://huggingface.co"
DEFAULT_WINDOWS_OUTPUT = Path("E:/DrivAerStar")
CASE_PATTERN = re.compile(r"^case_E/(\d{5})/case/([^/]+)$")
FILE_SUFFIXES = {
    "case": ".case",
    "geo": ".geo",
    "pressure": ".Pressure",
    "velocity": ".Velocity",
    "wss": ".WallShearStress",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_WINDOWS_OUTPUT)
    parser.add_argument("--start-case", type=int, default=1)
    parser.add_argument("--end-case", type=int, default=200)
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Limit selected cases for a test; zero means no limit.",
    )
    parser.add_argument(
        "--require-complete-range",
        action="store_true",
        help="Fail instead of skipping case IDs that do not exist in the repository.",
    )
    parser.add_argument(
        "--file-types",
        nargs="+",
        choices=tuple(FILE_SUFFIXES),
        default=list(FILE_SUFFIXES),
        help="Files to download for every selected case.",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT),
        help="Hugging Face endpoint. Keep the official endpoint for gated data.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Actually download files. Without this flag, only print the plan.",
    )
    parser.add_argument(
        "--zip-cases",
        action="store_true",
        help="After download, create case_E_XXXXX.zip for every complete case.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download again even if Hugging Face already has a cached copy.",
    )
    args = parser.parse_args()
    if args.start_case <= 0 or args.end_case < args.start_case:
        parser.error("Require 0 < --start-case <= --end-case")
    if args.max_cases < 0:
        parser.error("--max-cases must be zero or positive")
    return args


def resolve_token() -> str:
    token = get_token()
    if token:
        return token
    raise RuntimeError(
        "No Hugging Face token was found. Request access to the gated dataset "
        "and then run `hf auth login`, or set the HF_TOKEN environment variable."
    )


def discover_files(
    revision: str,
    endpoint: str,
    token: str,
    start_case: int,
    end_case: int,
    file_types: list[str],
) -> dict[int, list[str]]:
    """Discover requested files through the paginated repository-tree API."""
    print("Reading the paginated Hugging Face case_E file tree...")
    api = HfApi(endpoint=endpoint, token=token)
    requested_suffixes = {
        name: FILE_SUFFIXES[name] for name in file_types
    }
    discovered: dict[int, dict[str, str]] = {}
    entries = api.list_repo_tree(
        repo_id=REPO_ID,
        path_in_repo="case_E",
        recursive=True,
        expand=False,
        revision=revision,
        repo_type=REPO_TYPE,
        token=token,
    )
    for entry in entries:
        filename = getattr(entry, "path", "")
        match = CASE_PATTERN.fullmatch(filename)
        if match is None:
            continue
        case_id = int(match.group(1))
        if not start_case <= case_id <= end_case:
            continue
        for file_type, suffix in requested_suffixes.items():
            if filename.endswith(suffix):
                case_files = discovered.setdefault(case_id, {})
                if file_type in case_files:
                    raise RuntimeError(
                        f"Case {case_id:05d} has more than one {file_type} file"
                    )
                case_files[file_type] = filename

    incomplete = {
        case_id: sorted(set(file_types) - set(case_files))
        for case_id, case_files in discovered.items()
        if set(case_files) != set(file_types)
    }
    if incomplete:
        preview = ", ".join(
            f"{case_id:05d} missing {missing}"
            for case_id, missing in list(incomplete.items())[:10]
        )
        raise RuntimeError(f"Incomplete cases found: {preview}")

    return {
        case_id: [case_files[file_type] for file_type in file_types]
        for case_id, case_files in sorted(discovered.items())
    }


def print_plan(
    args: argparse.Namespace,
    cases: dict[int, list[str]],
    available_case_count: int,
) -> None:
    file_count = sum(len(files) for files in cases.values())
    print("=" * 72)
    print("DrivAerStar Hugging Face download plan")
    print(f"Repository       : {REPO_ID}")
    print(f"Revision         : {args.revision}")
    print(f"Endpoint         : {args.endpoint}")
    print(f"Requested range  : {args.start_case}-{args.end_case}")
    print(f"Available in range: {available_case_count}")
    print(f"Selected cases   : {len(cases)}")
    print(f"Selected files   : {file_count}")
    print(f"File types       : {', '.join(args.file_types)}")
    print(f"Output directory : {args.output_dir.expanduser()}")
    print(f"Create ZIPs      : {args.zip_cases}")
    print(f"Mode             : {'download' if args.download else 'preview only'}")
    if cases:
        case_ids = list(cases)
        print(f"First case IDs   : {case_ids[:10]}")
        print(f"Last case IDs    : {case_ids[-10:]}")
    print("=" * 72)


def archive_case(output_dir: Path, case_id: int, filenames: list[str]) -> Path:
    archive_dir = output_dir / "zip"
    archive_dir.mkdir(parents=True, exist_ok=True)
    destination = archive_dir / f"case_E_{case_id:05d}.zip"
    temporary = destination.with_suffix(".zip.part")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
        for filename in filenames:
            source = output_dir / filename
            if not source.is_file():
                raise FileNotFoundError(f"Cannot archive missing file: {source}")
            archive.write(source, arcname=filename)
    os.replace(temporary, destination)
    return destination


def download_cases(
    args: argparse.Namespace,
    token: str,
    cases: dict[int, list[str]],
) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    total_files = sum(len(files) for files in cases.values())
    completed = 0
    for case_position, (case_id, filenames) in enumerate(cases.items(), start=1):
        print(f"[case {case_position}/{len(cases)}] {case_id:05d}")
        for filename in filenames:
            completed += 1
            destination = output_dir / filename
            if destination.is_file() and not args.force_download:
                print(f"  [{completed}/{total_files}] exists: {filename}")
                continue
            print(f"  [{completed}/{total_files}] downloading: {filename}")
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                revision=args.revision,
                filename=filename,
                local_dir=output_dir,
                token=token,
                endpoint=args.endpoint,
                force_download=args.force_download,
            )
        if args.zip_cases:
            archive = archive_case(output_dir, case_id, filenames)
            print(f"  ZIP: {archive}")
    print(f"Download complete: {len(cases)} cases, {total_files} files")


def main() -> int:
    args = parse_args()
    token = resolve_token()
    try:
        cases = discover_files(
            args.revision,
            args.endpoint,
            token,
            args.start_case,
            args.end_case,
            args.file_types,
        )
    except (HfHubHTTPError, RequestException) as error:
        raise RuntimeError(
            "Unable to list the gated DrivAerStar repository. Confirm that the "
            "network is available and the logged-in Hugging Face account has "
            "been approved for access."
        ) from error
    requested_case_ids = set(range(args.start_case, args.end_case + 1))
    missing_case_ids = sorted(requested_case_ids - set(cases))
    if missing_case_ids and args.require_complete_range:
        preview = ", ".join(f"{case_id:05d}" for case_id in missing_case_ids[:20])
        raise FileNotFoundError(
            f"The requested range is missing {len(missing_case_ids)} case IDs: "
            f"{preview}. Remove --require-complete-range to skip them."
        )
    if missing_case_ids:
        preview = ", ".join(f"{case_id:05d}" for case_id in missing_case_ids[:20])
        suffix = " ..." if len(missing_case_ids) > 20 else ""
        print(
            f"WARNING: skipping {len(missing_case_ids)} case IDs that are absent "
            f"from the repository: {preview}{suffix}"
        )
    available_case_count = len(cases)
    if args.max_cases:
        cases = dict(list(cases.items())[: args.max_cases])
    if not cases:
        raise FileNotFoundError(
            f"No repository cases found in {args.start_case}-{args.end_case}"
        )
    print_plan(args, cases, available_case_count)
    if not args.download:
        print("Preview finished. Add --download to start downloading.")
        return 0
    download_cases(args, token, cases)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
