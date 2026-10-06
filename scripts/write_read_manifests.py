"""Record what PyNWB reads from each Zarr v3 store, as a JSON manifest per store.

The manifests are the expected values for ``scripts/verifyStoreContents.m``, which reads
the same stores with MatNWB and compares. For every store this records:

- ``objects``: each typed group and dataset, with its neurodata type and object id;
- ``links``: each soft or external link and the node it points to;
- ``references``: each object reference held in an attribute or a reference dataset;
- ``datasets``: shape, element kind, sum and sampled elements of each dataset, and for
  larger numeric datasets the sum over one block that spans a chunk boundary;
- ``compounds``: sampled records of each compound dataset, with references as paths;
- ``attributes``: the value of each schema attribute;
- ``tables``: column names, row count, ids and selected rows of each DynamicTable,
  with ragged columns resolved through their index and DynamicTableRegion columns
  as raw row indices.

Paths are absolute HDF5-style paths ("/acquisition/ts/data"); attribute paths append
the attribute name to the path of the node that holds it. Indices are zero-based and
in numpy (row-major) order.

Usage: python scripts/write_read_manifests.py STORE_DIR [--output DIR]
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import zarr
from hdmf.build import Builder, DatasetBuilder, GroupBuilder, ReferenceBuilder
from hdmf.common import DynamicTable, DynamicTableRegion, VectorIndex
from hdmf.container import AbstractContainer
from hdmf_zarr.nwb import NWBZarrIO


# A dataset with at most this many elements has every element recorded; a larger one
# has a fixed set of sampled positions instead.
MAX_FULL_ELEMENTS = 64
NUM_SPACED_SAMPLES = 6
NUM_NONZERO_SAMPLES = 4
# A dataset with more elements than this is never loaded whole: some tutorials write
# sparse arrays whose logical size runs to terabytes. Its sum is not recorded, and its
# samples are read one element at a time.
MAX_LOADED_ELEMENTS = 10_000_000
# Largest block read from a dataset to check a multi-element (and usually multi-chunk)
# read. Trailing axes are read whole when they hold at most MAX_FULL_TRAILING elements.
MAX_BLOCK_ELEMENTS = 1_000_000
MAX_FULL_TRAILING = 10_000
TRAILING_WINDOW = 100
# Table rows recorded from the start of each table, in addition to the last row.
NUM_LEADING_ROWS = 5

# Bookkeeping attributes that MatNWB keeps outside the schema properties, and the
# xarray dimension names hdmf-zarr adds to every array.
SKIPPED_ATTRIBUTES = frozenset({"namespace", "neurodata_type", "object_id", "_ARRAY_DIMENSIONS"})
# hdmf-zarr writes the cached specifications here; MatNWB generates classes from them
# but does not expose them as properties.
SPECIFICATIONS_PATH = "/specifications"

ISO_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")


def node_path(builder: Builder) -> str:
    """Absolute path of a builder, without hdmf's "root" prefix."""
    parts = builder.path.split("/")[1:]
    return "/" + "/".join(parts)


def to_json_scalar(value: Any) -> Any:
    """Convert one element to a JSON value, keeping non-finite floats as strings."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value[()]
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        return value
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    return str(value)


def element_kind(values: np.ndarray) -> str:
    """Classify an array's elements as numeric, bool, text or datetime."""
    if values.dtype.kind == "b":
        return "bool"
    if values.dtype.kind in "iuf":
        return "numeric"
    flat = values.reshape(-1)
    if flat.size and all(isinstance(v, str) and ISO_DATETIME.match(v) for v in flat):
        return "datetime"
    return "text"


def posix_seconds(text: str) -> float:
    """Seconds since the epoch for an ISO 8601 timestamp; naive timestamps are UTC."""
    moment = datetime.datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=datetime.timezone.utc)
    return moment.timestamp()


def encode_element(value: Any, kind: str) -> Any:
    if kind == "datetime":
        return posix_seconds(str(value))
    return to_json_scalar(value)


def sample_indices(shape: tuple[int, ...], values: np.ndarray) -> list[tuple[int, ...]]:
    """Positions to record: every element of a small array, else spaced and nonzero ones."""
    count = int(np.prod(shape))
    if count <= MAX_FULL_ELEMENTS:
        linear = list(range(count))
    else:
        linear = sorted({0, count - 1, *np.linspace(0, count - 1, NUM_SPACED_SAMPLES, dtype=int).tolist()})
        if values.dtype.kind in "iufb":
            nonzero = np.flatnonzero(values.reshape(-1))
            linear = sorted(set(linear) | set(nonzero[:NUM_NONZERO_SAMPLES].tolist()))
    return [tuple(int(i) for i in np.unravel_index(index, shape)) for index in linear]


def block_window(shape: tuple[int, ...], chunks: tuple[int, ...] | None) -> list[slice]:
    """A block of at most MAX_BLOCK_ELEMENTS that straddles a chunk boundary of axis 0."""
    trailing = [n if math.prod(shape[1:]) <= MAX_FULL_TRAILING else min(n, TRAILING_WINDOW) for n in shape[1:]]
    length = min(shape[0], max(1, MAX_BLOCK_ELEMENTS // max(1, math.prod(trailing))))
    if chunks and chunks[0] < shape[0]:
        boundary = max(chunks[0], (shape[0] // 2 // chunks[0]) * chunks[0])
        start = boundary - length // 2
    else:
        start = (shape[0] - length) // 2
    start = min(max(start, 0), shape[0] - length)
    window = [slice(start, start + length)]
    window += [slice((n - w) // 2, (n - w) // 2 + w) for n, w in zip(shape[1:], trailing)]
    return window


def describe_block(array: Any, shape: tuple[int, ...], kind: str) -> dict[str, Any] | None:
    """Sum over one block of a numeric dataset that is too large to record in full."""
    if kind not in ("numeric", "bool") or not shape or math.prod(shape) <= MAX_FULL_ELEMENTS:
        return None
    chunks = getattr(array, "chunks", None)
    window = block_window(shape, tuple(chunks) if chunks else None)
    block = np.asarray(array[tuple(window)]).astype(np.float64)
    return {
        "start": [w.start for w in window],
        "stop": [w.stop for w in window],
        "count": int(block.size),
        "sum": float(np.sum(block[np.isfinite(block)])),
    }


def describe_large_array(path: str, array: zarr.Array) -> dict[str, Any]:
    """Describe an array too large to load, from point reads and its written chunks."""
    shape = tuple(int(n) for n in array.shape)
    count = int(np.prod(shape))
    kind = element_kind(np.zeros(0, dtype=array.dtype))
    linear = {0, count - 1, *np.linspace(0, count - 1, NUM_SPACED_SAMPLES, dtype=np.int64).tolist()}
    indices = {tuple(int(i) for i in np.unravel_index(index, shape)) for index in linear}
    indices |= set(nonzero_indices_in_written_chunks(array))
    return {
        "path": path,
        "kind": kind,
        "shape": list(shape),
        "count": count,
        "sum": None,
        "block": describe_block(array, shape, kind),
        "samples": [
            {"index": list(index), "value": encode_element(array[index], kind)}
            for index in sorted(indices)
        ],
    }


def nonzero_indices_in_written_chunks(array: zarr.Array) -> list[tuple[int, ...]]:
    """Up to NUM_NONZERO_SAMPLES nonzero positions, found by reading written chunks only."""
    chunk_shape = array.chunks
    found: list[tuple[int, ...]] = []
    chunk_root = Path(array.store.root) / array.path / "c"
    for chunk_file in sorted(p for p in chunk_root.rglob("*") if p.is_file()):
        grid = tuple(int(part) for part in chunk_file.relative_to(chunk_root).parts)
        origin = tuple(g * c for g, c in zip(grid, chunk_shape))
        region = tuple(slice(o, min(o + c, n)) for o, c, n in zip(origin, chunk_shape, array.shape))
        block = np.asarray(array[region])
        for offset in np.argwhere(block != 0)[: NUM_NONZERO_SAMPLES - len(found)]:
            found.append(tuple(int(o + i) for o, i in zip(origin, offset)))
        if len(found) >= NUM_NONZERO_SAMPLES:
            break
    return found


def describe_array(path: str, data: Any) -> dict[str, Any]:
    chunks = getattr(data, "chunks", None)
    values = np.asarray(data[()] if hasattr(data, "shape") else data)
    if values.dtype.kind == "O":
        values = np.vectorize(to_json_scalar, otypes=[object])(values) if values.size else values
    shape = tuple(int(n) for n in values.shape)
    kind = element_kind(values)
    entry: dict[str, Any] = {
        "path": path, "kind": kind, "shape": list(shape), "count": int(values.size), "sum": None, "block": None
    }
    if kind in ("numeric", "bool"):
        finite = values.astype(np.float64)
        entry["sum"] = float(np.sum(finite[np.isfinite(finite)]))
        entry["block"] = describe_block(_Chunked(values, chunks), shape, kind)
    entry["samples"] = [
        {"index": list(index), "value": encode_element(values[index], kind)}
        for index in sample_indices(shape, values)
    ]
    return entry


def attribute_value(value: Any) -> tuple[str, Any] | None:
    """Kind and JSON value of a plain attribute, or None for one that is not plain data."""
    if isinstance(value, (list, tuple, np.ndarray)):
        values = np.asarray(value)
        if values.dtype.kind == "O" and not all(isinstance(v, (str, bytes)) for v in values.reshape(-1)):
            return None
        kind = element_kind(values)
        return kind, [encode_element(v, kind) for v in values.reshape(-1)]
    if isinstance(value, (str, bytes, bool, int, float, np.generic)):
        values = np.asarray(to_json_scalar(value))
        kind = element_kind(values) if not isinstance(value, float) else "numeric"
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            kind = "numeric"
        return kind, encode_element(to_json_scalar(value), kind)
    return None


def target_record(target: Builder) -> dict[str, Any]:
    if isinstance(target, ReferenceBuilder):
        target = target.builder
    return {"path": node_path(target), "object_id": target.attributes.get("object_id")}


class ManifestWriter:
    def __init__(self, store_path: Path):
        self.store_path = store_path
        self.manifest: dict[str, Any] = {
            "store": store_path.name,
            "objects": [],
            "links": [],
            "references": [],
            "datasets": [],
            "compounds": [],
            "attributes": [],
            "tables": [],
        }

    def write(self, output_dir: Path) -> Path:
        with NWBZarrIO(str(self.store_path), "r") as io:
            nwbfile = io.read()
            root = io.read_builder()
            self.walk_group(root)
            for container in nwbfile.objects.values():
                if isinstance(container, DynamicTable):
                    self.record_table(io, container)

        output_path = output_dir / f"{self.store_path.name}.manifest.json"
        output_path.write_text(json.dumps(self.manifest, indent=1), encoding="utf-8")
        return output_path

    def walk_group(self, group: GroupBuilder) -> None:
        path = node_path(group)
        if path == SPECIFICATIONS_PATH:
            return
        self.record_node(path, group)
        link_records = self.stored_link_records(path) if group.links else {}
        for name, link in group.links.items():
            target = link.builder
            # hdmf builds the target of an external link without its parent chain,
            # so its builder path is wrong; the stored link record holds the path.
            target_path = "/" + link_records[name]["path"].lstrip("/")
            self.manifest["links"].append(
                {
                    "path": f"{path.rstrip('/')}/{name}",
                    "target": {"path": target_path, "object_id": target.attributes.get("object_id")},
                    "external": Path(target.source).resolve() != self.store_path.resolve(),
                    "source": Path(target.source).name,
                }
            )
        for dataset in group.datasets.values():
            self.walk_dataset(dataset)
        for subgroup in group.groups.values():
            self.walk_group(subgroup)

    def stored_link_records(self, path: str) -> dict[str, dict[str, Any]]:
        """The group's link records as hdmf-zarr stored them, keyed by link name."""
        group = zarr.open_group(str(self.store_path), mode="r", path=path.lstrip("/"))
        return {record["name"]: record for record in group.attrs.get("_LINKS", [])}

    def record_node(self, path: str, builder: Builder) -> None:
        attributes = builder.attributes
        if "neurodata_type" in attributes:
            self.manifest["objects"].append(
                {
                    "path": path,
                    "neurodata_type": attributes["neurodata_type"],
                    "namespace": attributes.get("namespace"),
                    "object_id": attributes.get("object_id"),
                }
            )
        for name, value in attributes.items():
            if name in SKIPPED_ATTRIBUTES:
                continue
            attribute_path = f"{path.rstrip('/')}/{name}"
            if isinstance(value, (Builder, ReferenceBuilder)):
                self.manifest["references"].append(
                    {"path": attribute_path, "kind": "attribute", "targets": [target_record(value)]}
                )
                continue
            described = attribute_value(value)
            if described is not None:
                kind, encoded = described
                self.manifest["attributes"].append({"path": attribute_path, "kind": kind, "value": encoded})

    def walk_dataset(self, dataset: DatasetBuilder) -> None:
        path = node_path(dataset)
        self.record_node(path, dataset)
        data = dataset.data
        if isinstance(dataset.dtype, list):
            self.record_compound(path, dataset)
        elif _holds_references(dataset):
            self.manifest["references"].append(
                {"path": path, "kind": "dataset", "targets": [target_record(b) for b in data]}
            )
        elif _element_count(data) > MAX_LOADED_ELEMENTS:
            array = zarr.open_array(store=str(self.store_path), path=path.lstrip("/"), mode="r")
            self.manifest["datasets"].append(describe_large_array(path, array))
        else:
            self.manifest["datasets"].append(describe_array(path, data))

    def record_compound(self, path: str, dataset: DatasetBuilder) -> None:
        field_names = [field["name"] for field in dataset.dtype]
        rows = list(dataset.data)
        count = len(rows)
        indices = sorted({*range(min(count, NUM_LEADING_ROWS)), count - 1}) if count else []
        records = []
        for index in indices:
            values = {}
            for name, value in zip(field_names, rows[index]):
                if isinstance(value, (Builder, ReferenceBuilder)):
                    values[name] = {"reference": target_record(value)}
                elif isinstance(value, AbstractContainer):
                    values[name] = {"reference": {"path": None, "object_id": value.object_id}}
                else:
                    values[name] = to_json_scalar(value)
            records.append({"index": index, "values": values})
        self.manifest["compounds"].append(
            {"path": path, "fields": field_names, "count": count, "records": records}
        )

    def record_table(self, io: NWBZarrIO, table: DynamicTable) -> None:
        builder = io.manager.get_builder(table)
        path = node_path(builder)
        count = len(table)
        indices = sorted({*range(min(count, NUM_LEADING_ROWS)), count - 1}) if count else []
        columns = {}
        for name in table.colnames:
            column = table[name]
            kind = _column_kind(column)
            columns[name] = kind
        rows = []
        for index in indices:
            values = {}
            for name in table.colnames:
                if columns[name] == "compound":
                    continue
                values[name] = _row_value(table[name], index)
            rows.append({"index": index, "values": values})
        self.manifest["tables"].append(
            {
                "path": path,
                "neurodata_type": builder.attributes.get("neurodata_type"),
                "colnames": list(table.colnames),
                "column_kinds": columns,
                "nrows": count,
                "ids": [int(i) for i in table.id.data[:]],
                "rows": rows,
            }
        )


class _Chunked:
    """An in-memory array that still reports the chunk shape it was stored with."""

    def __init__(self, values: np.ndarray, chunks: Any):
        self.values = values
        self.chunks = chunks

    def __getitem__(self, key: Any) -> np.ndarray:
        return self.values[key]


def _element_count(data: Any) -> int:
    shape = getattr(data, "shape", None)
    return int(np.prod(shape)) if shape is not None else 1


def _holds_references(dataset: DatasetBuilder) -> bool:
    data = dataset.data
    if dataset.dtype == "object":
        return True
    try:
        first = data[0] if len(data) else None
    except TypeError:
        return False
    return isinstance(first, (Builder, ReferenceBuilder))


def _column_kind(column: Any) -> str:
    """How a column's row values are recorded: ragged, region, reference, compound or plain.

    Compound columns (also ragged ones, such as pixel_mask) are not recorded per row:
    their datasets are checked through the manifest's compounds list instead.
    """
    target = column
    while isinstance(target, VectorIndex):
        target = target.target
    data = target.data
    if len(data) and isinstance(data[0], (tuple, np.void)):
        return "compound"
    if isinstance(column, VectorIndex):
        return "ragged"
    if isinstance(column, DynamicTableRegion):
        return "region"
    if len(data) and isinstance(data[0], AbstractContainer):
        return "reference"
    return "plain"


def _row_value(column: Any, index: int) -> Any:
    if isinstance(column, VectorIndex):
        start = 0 if index == 0 else int(column.data[index - 1])
        stop = int(column.data[index])
        target = column.target
        if isinstance(target, VectorIndex):
            return [_row_value(target, i) for i in range(start, stop)]
        # One slice read: element-by-element reads would decompress a chunk each.
        return [_plain_value(value) for value in target.data[start:stop]]
    if isinstance(column, DynamicTableRegion):
        return int(column.data[index])
    return _plain_value(column.data[index])


def _plain_value(value: Any) -> Any:
    if isinstance(value, AbstractContainer):
        return {"object_id": value.object_id}
    if isinstance(value, np.ndarray):
        return [_plain_value(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_plain_value(v) for v in value]
    return to_json_scalar(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("store_dir", help="folder holding the *.nwb.zarr stores")
    parser.add_argument("--output", metavar="DIR", help="folder to write manifests to (default: STORE_DIR)")
    arguments = parser.parse_args()

    store_dir = Path(arguments.store_dir).resolve()
    output_dir = Path(arguments.output).resolve() if arguments.output else store_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    failures = {}
    for store_path in sorted(store_dir.glob("*.nwb.zarr")):
        try:
            output_path = ManifestWriter(store_path).write(output_dir)
            print(f"  wrote {output_path.name}")
        except Exception as error:  # report every store, then fail
            failures[store_path.name] = f"{type(error).__name__}: {error}"
            print(f"  FAIL  {store_path.name}: {failures[store_path.name]}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
