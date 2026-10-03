"""Download real NWB files from DANDI and convert them to Zarr v3 stores with hdmf-zarr.

No real NWB data is published as Zarr v3 yet, so the real-data check converts HDF5
files: PyNWB reads each one and hdmf-zarr exports it, which is how existing data will
reach Zarr v3 in practice. Each file is pinned to an asset of a published Dandiset
version and verified against its SHA-256 digest before conversion.

Usage: python scripts/convert_real_datasets.py --cache DIR --output DIR
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import sys
import urllib.request

from hdmf_zarr.nwb import NWBZarrIO
from pynwb import NWBHDF5IO


# Store name -> pinned DANDI asset.
REAL_DATASETS: dict[str, dict[str, str]] = {
    # Huszár et al., "Preconfigured dynamics in the hippocampus are guided by embryonic
    # birthdate and rate of neurogenesis", CC-BY-4.0. 2.5 GB: a 21.9M x 64 int16 LFP
    # ElectricalSeries, 64 electrodes and 167 units with ragged spike times.
    "sub-e16-3m1_ses-e16-3m1-210201.nwb.zarr": {
        "dandiset": "000552",
        "version": "0.230630.2304",
        "path": "sub-e16-3m1/sub-e16-3m1_ses-e16-3m1-210201_behavior+ecephys.nwb",
        "asset_id": "75a0be1f-61da-4377-bcd0-42dce58cc810",
        "sha256": "b495df7c1d06231ad6633f097d4159d9d39a07301e882590c03dcb0b86d71395",
    },
}

DOWNLOAD_URL = "https://api.dandiarchive.org/api/assets/{asset_id}/download/"
READ_BLOCK_BYTES = 16 * 1024 * 1024


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(READ_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def fetch(asset: dict[str, str], cache_dir: Path) -> Path:
    """Path of the verified HDF5 file, downloading it unless the cache holds it."""
    target = cache_dir / f"{asset['asset_id']}.nwb"
    if target.is_file() and sha256_of(target) == asset["sha256"]:
        print(f"  using cached {target.name}", flush=True)
        return target

    print(f"  downloading {asset['path']} from Dandiset {asset['dandiset']}", flush=True)
    partial = target.with_suffix(".part")
    with urllib.request.urlopen(DOWNLOAD_URL.format(asset_id=asset["asset_id"])) as response:
        with partial.open("wb") as file:
            shutil.copyfileobj(response, file, READ_BLOCK_BYTES)
    digest = sha256_of(partial)
    if digest != asset["sha256"]:
        partial.unlink()
        raise RuntimeError(f"SHA-256 mismatch for {asset['path']}: expected {asset['sha256']}, got {digest}")
    partial.rename(target)
    return target


def convert(source: Path, store_path: Path) -> None:
    if store_path.exists():
        shutil.rmtree(store_path)
    with NWBHDF5IO(str(source), "r", load_namespaces=True) as read_io:
        with NWBZarrIO(str(store_path), mode="w") as export_io:
            # The data is copied rather than linked: a Zarr store cannot link into
            # an HDF5 file.
            export_io.export(src_io=read_io, write_args={"link_data": False})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", required=True, metavar="DIR", help="folder for the downloaded HDF5 files")
    parser.add_argument("--output", required=True, metavar="DIR", help="folder to write the Zarr v3 stores to")
    arguments = parser.parse_args()

    cache_dir = Path(arguments.cache).resolve()
    output_dir = Path(arguments.output).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for store_name, asset in REAL_DATASETS.items():
        print(f"Preparing {store_name}", flush=True)
        source = fetch(asset, cache_dir)
        convert(source, output_dir / store_name)
        print(f"  wrote {store_name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
