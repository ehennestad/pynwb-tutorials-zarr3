# PyNWB tutorials on Zarr v3

The [PyNWB](https://github.com/NeurodataWithoutBorders/pynwb) gallery tutorials that write an
NWB file, rewritten to read and write **Zarr format 3** stores instead of HDF5 files.

Each tutorial is a standalone script. Where the upstream tutorial used
`pynwb.NWBHDF5IO` and a `*.nwb` file, these use `hdmf_zarr.nwb.NWBZarrIO` and a `*.nwb.zarr`
store; `H5DataIO` is replaced by `ZarrDataIO` with `numcodecs` compressors. Everything else --
the neurodata types built, the narrative, the section structure -- follows upstream.

## Requirements

Zarr format 3 output is **not available in any released `hdmf-zarr`**: the latest release
(0.13.0) pins `zarr<3.0`, which writes Zarr format 2. Format 3 requires the unreleased
[`zarr-v3-migration`](https://github.com/hdmf-dev/hdmf-zarr/pull/325) branch, which
`pyproject.toml` pins directly. Expect this pin to be replaced by a normal version
constraint once that PR is released.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

## Running

Run every tutorial and verify each generated store declares `zarr_format: 3`:

```bash
.venv/bin/python scripts/run_zarr_v3_tutorials.py
```

Each tutorial runs in a throwaway temporary directory. To keep the stores around, or to run
only some tutorials:

```bash
.venv/bin/python scripts/run_zarr_v3_tutorials.py --keep-output out domain/ecephys.py
```

Tutorials are also plain scripts and can be run directly; they write into the current
working directory:

```bash
.venv/bin/python tutorials/domain/ecephys.py
```

## Tutorials

| Tutorial | Output store(s) |
| --- | --- |
| `general/plot_file.py` | `basics_tutorial.nwb.zarr` |
| `general/scratch.py` | `raw_data`, `processed_data`, `scratch_analysis` |
| `general/plot_timeintervals.py` | `example_timeintervals_file` |
| `general/add_remove_containers.py` | `nwbfile`, `exported_nwbfile`, `legacy_device_model`, `upgraded_device_model` |
| `general/extensions.py` | `cache_spec_example`, `test_multicontainerinterface`, `test_cortical_surface` |
| `general/plot_external_resources.py` | `external_resources_tutorial` |
| `domain/ecephys.py` | `ecephys_tutorial` |
| `domain/ophys.py` | `ophys_tutorial` |
| `domain/plot_behavior.py` | `behavioral_tutorial` |
| `domain/plot_icephys.py` | `ex_test_icephys_file`, `test_icephys_file` |
| `domain/images.py` | `images_tutorial` |
| `advanced_io/plot_zarr_io.py` | `zarr_tutorial` |
| `advanced_io/plot_iterative_write.py` | `basic_iterwrite_example` plus six `basic_sparse_iterwrite_*` stores |
| `advanced_io/plot_linking_data.py` | `external1_example`, `external2_example`, `external_linkcontainer_example`, `external_linkdataset_example` |

All output names carry the `.nwb.zarr` suffix.

## Cross-backend check: reading the stores with MatNWB

A [workflow](.github/workflows/matnwb-read.yml) writes the stores with PyNWB and then opens every
one of them with [MatNWB](https://github.com/NeurodataWithoutBorders/matnwb)'s Zarr v3 reader
(branch `zarr-support/5-zarr3-reader`, which pulls in `zarr-matlab` and `hdmf-zarr-matlab`). It
runs on push, on pull requests, and weekly — both sides track moving branches, so most breakage
will arrive from upstream rather than from a change here.

Stores MatNWB cannot yet read are listed with reasons in
[`ci/matnwb_known_failures.txt`](ci/matnwb_known_failures.txt). The check fails if a store outside
that list fails, and also if a listed store starts passing, so stale entries get removed rather
than quietly masking a fix.

To run it locally, with MatNWB and its requirements on your MATLAB path:

```matlab
verifyStoresReadable("stores", KnownFailureFile="ci/matnwb_known_failures.txt")
```

As of the last run, 29 of 31 stores read successfully. The two exceptions:

| Store | Cause |
| --- | --- |
| `legacy_device_model.nwb.zarr` | Expected. The store is a deliberately synthesised pre-2.9 file with a string `Device.model`; PyNWB upgrades it on read, MatNWB has no such upgrade. |
| `processed_data.nwb.zarr` | The store is readable. MatNWB resolves the relative external-link path against the process working directory instead of the directory containing the store, so `../raw_data.nwb.zarr` is looked up in the wrong place. |

Two details matter for reproducing this outside CI. `nwbRead` is called **without**
`ignorecache`, because several stores embed extension schemas (`mylab`, `ecog`,
`test_multicontainerinterface`) whose classes exist only once generated from the store
being read. And the generated-class folder is emptied first: a populated folder, or a
MatNWB checkout still holding classes from earlier work, makes those stores appear to
read on a machine where a clean checkout would fail.

## What changed relative to upstream

Most edits are mechanical: `NWBHDF5IO` → `NWBZarrIO`, `*.nwb` → `*.nwb.zarr`, `H5DataIO` →
`ZarrDataIO`, and prose that said "file" where it now means "store". `force_overwrite=True` is
passed on write so a tutorial can be re-run over an existing store. Three changes are worth
calling out:

- **Compressors are Zarr v3 codecs.** `compression="gzip", compression_opts=4` becomes
  `compressor=GzipCodec(level=4)`; a Blosc compressor comes from `zarr.codecs.BloscCodec` rather
  than `numcodecs.Blosc`.
- **Zarr compresses by default.** zarr-python 3 applies Zstd unless told otherwise, so
  `plot_iterative_write.py` passes `compressor=False` for the cases the tutorial presents as
  uncompressed. Without that opt-out its chunk-size comparison collapses — the 80 MB
  large-chunk case comes out under 1 MB and the point of the section disappears.
- **Store size is a directory size.** `plot_iterative_write.py` sums the files under the store
  directory instead of calling `os.stat().st_size` on a single file.

## Sections omitted from otherwise-converted tutorials

| Tutorial | Omitted | Reason |
| --- | --- | --- |
| `advanced_io/plot_iterative_write.py` | "Alternative Approach: User-defined dataset write" | `ZarrDataIO` takes only `data`, `chunks`, `fillvalue`, `compressor`, `filters` and `link_data`. It cannot declare a dataset by `shape`/`dtype` and allocate it empty, and Zarr arrays have no `maxshape`. |
| `advanced_io/plot_linking_data.py` | "Automatically splitting large data across multiple HDF5 files" | Relies on the h5py `family` driver and on empty-dataset allocation. A Zarr store is already split across many chunk files, so the problem does not arise. |
| `general/plot_external_resources.py` | The `HERD.get_object_entities` call | hdmf-zarr does not preserve the integer index fields of HERD's compound datasets on round-trip, so `objects.files_idx` reads back as `float64` and the row lookup rejects it. Verified on both hdmf-zarr 0.13.0 (zarr 2, where the compound dataset comes back as `object` dtype) and the `zarr-v3-migration` branch — a general Zarr-backend limitation, not a Zarr v3 regression. |

Each omission is replaced in-file by a note explaining what was dropped and why.

## Tutorials deliberately excluded

| Upstream tutorial | Reason |
| --- | --- |
| `general/plot_read_basics.py`, `advanced_io/streaming.py`, `general/resources_streaming.py` | Read files downloaded or streamed from the DANDI Archive; they write nothing. |
| `general/object_id.py`, `general/plot_configurator.py`, `domain/ogen.py` | Produce no output file. |
| `advanced_io/parallelio.py` | Entire example is commented-out MPI/parallel-HDF5 code that writes nothing, and parallel HDF5 has no Zarr analogue. |
| `advanced_io/h5dataio.py` | Subject matter is HDF5-specific dataset I/O (`H5DataIO`). The Zarr equivalent is covered by `plot_zarr_io.py`. |
| `advanced_io/plot_editing.py` | Three of its sections edit the file through raw `h5py` calls (renaming, changing dtype and shape) that have no clean Zarr analogue. |
| `domain/plot_icephys_pandas.py` | Writes through `pynwb.testing.icephys_testutils.create_icephys_testfile`, which hardcodes `NWBHDF5IO`. |

## Provenance

Tutorials were taken from PyNWB `dev` at commit `6285dd8c` and modified only where the storage
backend required it. The upstream tutorials are BSD-3-Clause licensed; see
[pynwb/license.txt](https://github.com/NeurodataWithoutBorders/pynwb/blob/dev/license.txt).
