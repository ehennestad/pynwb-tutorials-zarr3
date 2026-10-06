"""Write NWB Zarr v3 stores that use storage options the tutorials leave at their defaults.

The tutorials write every array with hdmf-zarr's defaults: no sharding, zstd, small
chunks, consolidated metadata. These stores cover the options a user can choose
through ``ZarrDataIO`` and ``NWBZarrIO.write``, so the content check also exercises
the reader paths they lead to:

- ``variant_codecs``: blosc with lz4 and bitshuffle, zstd followed by a crc32c
  checksum, a transpose filter, a big-endian serializer, and float32, int32, int8 and
  uint32 data.
- ``variant_sharded``: sharded arrays, read through the shard index and Range
  requests, including a 1-D array whose last shard is partial.
- ``variant_lz4``: a numcodecs LZ4 compressor, which zarr-matlab does not implement.
- ``variant_unconsolidated``: written with ``consolidate_metadata=False``, so a reader
  has to list the store to browse it, which an HTTP server cannot do.
- ``variant_appended``: written, then reopened with mode "a" and a TimeSeries added.
- ``variant_appended_unconsolidated``: the same append with ``consolidate_metadata=False``,
  which leaves the consolidated metadata in the root describing the file before the
  append. zarr-python, and through it PyNWB, browses a store through that metadata
  when it is present, as MatNWB does, so both read the file without the appended
  TimeSeries and the check confirms that they agree.

Usage: python scripts/write_format_variants.py OUTPUT_DIR
"""

from __future__ import annotations

import datetime
from pathlib import Path
import shutil
import sys
from typing import Any, Callable

import numpy as np
from hdmf_zarr import ZarrDataIO
from hdmf_zarr.nwb import NWBZarrIO
from pynwb import NWBFile, TimeSeries
import zarr.codecs
import zarr.codecs.numcodecs

SESSION_START = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
RANDOM = np.random.default_rng(seed=0)


def new_nwbfile(identifier: str) -> NWBFile:
    return NWBFile(
        session_description=f"Zarr v3 storage variant: {identifier}",
        identifier=identifier,
        session_start_time=SESSION_START,
    )


def add_series(nwbfile: NWBFile, name: str, data: Any) -> None:
    nwbfile.add_acquisition(TimeSeries(name=name, data=data, unit="a.u.", rate=1000.0))


def write(path: Path, nwbfile: NWBFile, **write_args: Any) -> None:
    if path.exists():
        shutil.rmtree(path)
    with NWBZarrIO(str(path), mode="w") as io:
        io.write(nwbfile, **write_args)


def append(path: Path, name: str, data: np.ndarray, **write_args: Any) -> None:
    with NWBZarrIO(str(path), mode="a") as io:
        nwbfile = io.read()
        add_series(nwbfile, name, data)
        io.write(nwbfile, **write_args)


def write_codecs(path: Path) -> None:
    nwbfile = new_nwbfile("variant_codecs")
    signal = RANDOM.standard_normal((5000, 4))
    add_series(nwbfile, "blosc_lz4_bitshuffle", ZarrDataIO(
        signal.astype(np.float32), chunks=(1000, 4),
        compressors=zarr.codecs.BloscCodec(cname="lz4", shuffle="bitshuffle", typesize=4),
    ))
    add_series(nwbfile, "zstd_crc32c", ZarrDataIO(
        signal, chunks=(1000, 4),
        compressors=[zarr.codecs.ZstdCodec(level=3), zarr.codecs.Crc32cCodec()],
    ))
    add_series(nwbfile, "transposed", ZarrDataIO(
        signal, chunks=(1000, 4), filters=[zarr.codecs.TransposeCodec(order=(1, 0))],
    ))
    add_series(nwbfile, "big_endian", ZarrDataIO(
        signal, chunks=(1000, 4), serializer=zarr.codecs.BytesCodec(endian="big"),
    ))
    add_series(nwbfile, "float32", signal.astype(np.float32))
    add_series(nwbfile, "int32", RANDOM.integers(-2**31, 2**31 - 1, size=(5000, 4), dtype=np.int32))
    add_series(nwbfile, "int8", RANDOM.integers(-128, 127, size=5000, dtype=np.int8))
    add_series(nwbfile, "uint32", RANDOM.integers(0, 2**32 - 1, size=5000, dtype=np.uint32))
    write(path, nwbfile)


def write_sharded(path: Path) -> None:
    nwbfile = new_nwbfile("variant_sharded")
    add_series(nwbfile, "sharded_2d", ZarrDataIO(
        RANDOM.standard_normal((100_000, 8)).astype(np.float32),
        chunks=(1000, 8), shards=(10_000, 8),
    ))
    # 25,500 elements in shards of 10,000: the last shard holds 5,500.
    add_series(nwbfile, "sharded_partial_edge", ZarrDataIO(
        np.arange(25_500, dtype=np.int64), chunks=(500,), shards=(10_000,),
    ))
    write(path, nwbfile)


def write_lz4(path: Path) -> None:
    nwbfile = new_nwbfile("variant_lz4")
    add_series(nwbfile, "lz4", ZarrDataIO(
        RANDOM.standard_normal(5000), chunks=(1000,), compressors=zarr.codecs.numcodecs.LZ4(),
    ))
    write(path, nwbfile)


def write_unconsolidated(path: Path) -> None:
    nwbfile = new_nwbfile("variant_unconsolidated")
    add_series(nwbfile, "signal", RANDOM.standard_normal((2000, 2)))
    write(path, nwbfile, consolidate_metadata=False)


def write_appended(path: Path) -> None:
    nwbfile = new_nwbfile("variant_appended")
    add_series(nwbfile, "original", RANDOM.standard_normal(2000))
    write(path, nwbfile)
    append(path, "appended", RANDOM.standard_normal(2000))


def write_appended_unconsolidated(path: Path) -> None:
    nwbfile = new_nwbfile("variant_appended_unconsolidated")
    add_series(nwbfile, "original", RANDOM.standard_normal(2000))
    write(path, nwbfile)
    append(path, "appended", RANDOM.standard_normal(2000), consolidate_metadata=False)


VARIANTS: dict[str, Callable[[Path], None]] = {
    "variant_codecs.nwb.zarr": write_codecs,
    "variant_sharded.nwb.zarr": write_sharded,
    "variant_lz4.nwb.zarr": write_lz4,
    "variant_unconsolidated.nwb.zarr": write_unconsolidated,
    "variant_appended.nwb.zarr": write_appended,
    "variant_appended_unconsolidated.nwb.zarr": write_appended_unconsolidated,
}


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    output_dir = Path(sys.argv[1]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for store_name, writer in VARIANTS.items():
        writer(output_dir / store_name)
        print(f"  wrote {store_name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
