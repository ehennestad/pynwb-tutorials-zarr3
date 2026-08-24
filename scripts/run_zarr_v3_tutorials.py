"""Run every tutorial and verify each one produced Zarr v3 stores.

Tutorials write their output into the working directory, so each run happens inside a
temporary directory that is discarded afterwards. Pass ``--keep-output DIR`` to write the
stores somewhere durable instead.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TUTORIALS_ROOT = REPOSITORY_ROOT / "tutorials"

# Tutorial path (relative to tutorials/) -> the stores it is expected to produce.
TUTORIAL_OUTPUTS: dict[str, tuple[str, ...]] = {
    "general/plot_file.py": ("basics_tutorial.nwb.zarr",),
    "general/scratch.py": (
        "raw_data.nwb.zarr",
        "processed_data.nwb.zarr",
        "scratch_analysis.nwb.zarr",
    ),
    "general/plot_timeintervals.py": ("example_timeintervals_file.nwb.zarr",),
    "general/add_remove_containers.py": (
        "nwbfile.nwb.zarr",
        "exported_nwbfile.nwb.zarr",
        "legacy_device_model.nwb.zarr",
        "upgraded_device_model.nwb.zarr",
    ),
    "general/extensions.py": (
        "cache_spec_example.nwb.zarr",
        "test_multicontainerinterface.nwb.zarr",
        "test_cortical_surface.nwb.zarr",
    ),
    "general/plot_external_resources.py": ("external_resources_tutorial.nwb.zarr",),
    "domain/ecephys.py": ("ecephys_tutorial.nwb.zarr",),
    "domain/ophys.py": ("ophys_tutorial.nwb.zarr",),
    "domain/plot_behavior.py": ("behavioral_tutorial.nwb.zarr",),
    "domain/plot_icephys.py": (
        "ex_test_icephys_file.nwb.zarr",
        "test_icephys_file.nwb.zarr",
    ),
    "domain/images.py": ("images_tutorial.nwb.zarr",),
    "advanced_io/plot_zarr_io.py": ("zarr_tutorial.nwb.zarr",),
    "advanced_io/plot_iterative_write.py": (
        "basic_iterwrite_example.nwb.zarr",
        "basic_sparse_iterwrite_example.nwb.zarr",
        "basic_sparse_iterwrite_compressed_example.nwb.zarr",
        "basic_sparse_iterwrite_largechunks_example.nwb.zarr",
        "basic_sparse_iterwrite_largechunks_compressed_example.nwb.zarr",
        "basic_sparse_iterwrite_largearray.nwb.zarr",
        "basic_sparse_iterwrite_multifile.nwb.zarr",
    ),
    "advanced_io/plot_linking_data.py": (
        "external1_example.nwb.zarr",
        "external2_example.nwb.zarr",
        "external_linkcontainer_example.nwb.zarr",
        "external_linkdataset_example.nwb.zarr",
    ),
}


def verify_zarr_v3(store_path: Path) -> None:
    """Raise if the store is missing or its root metadata does not declare Zarr format 3."""
    metadata_path = store_path / "zarr.json"
    if not metadata_path.is_file():
        raise AssertionError(f"missing root metadata: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("zarr_format") != 3:
        raise AssertionError(
            f"expected Zarr format 3 in {metadata_path}, found {metadata.get('zarr_format')!r}"
        )


def run_tutorial(tutorial: str, outputs: tuple[str, ...], work_dir: Path) -> str | None:
    """Run one tutorial in ``work_dir`` and return a failure description, or None on success."""
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"

    result = subprocess.run(
        [sys.executable, str(TUTORIALS_ROOT / tutorial)],
        cwd=work_dir,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        return f"exited with status {result.returncode}\n{output.rstrip()}"

    try:
        for output in outputs:
            verify_zarr_v3(work_dir / output)
    except (AssertionError, json.JSONDecodeError, OSError) as error:
        return f"output verification failed: {error}"

    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-output",
        metavar="DIR",
        help="write the generated stores into DIR and keep them instead of using a temporary directory",
    )
    parser.add_argument(
        "tutorials",
        nargs="*",
        help="tutorial paths relative to tutorials/ (default: all of them)",
    )
    arguments = parser.parse_args()

    selected = arguments.tutorials or list(TUTORIAL_OUTPUTS)
    unknown = [tutorial for tutorial in selected if tutorial not in TUTORIAL_OUTPUTS]
    if unknown:
        parser.error(f"unknown tutorial(s): {', '.join(unknown)}")

    if arguments.keep_output:
        work_dir = Path(arguments.keep_output).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        temporary_dir = None
    else:
        temporary_dir = tempfile.mkdtemp(prefix="pynwb-tutorials-zarr3-")
        work_dir = Path(temporary_dir)

    failures: dict[str, str] = {}
    try:
        for tutorial in selected:
            print(f"Running {tutorial} ...", flush=True)
            failure = run_tutorial(tutorial, TUTORIAL_OUTPUTS[tutorial], work_dir)
            if failure is None:
                print("  PASS")
            else:
                failures[tutorial] = failure
                print("  FAIL")
    finally:
        if temporary_dir is not None:
            shutil.rmtree(temporary_dir, ignore_errors=True)

    print("\nSummary")
    print(f"  Passed: {len(selected) - len(failures)}")
    print(f"  Failed: {len(failures)}")
    for tutorial, failure in failures.items():
        print(f"\n--- {tutorial} ---\n{failure}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
