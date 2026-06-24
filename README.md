# RowPack

![Training throughput](docs/images/mm_infographic_vqa_samples_per_s.png)

![Input file size](docs/images/mm_infographic_vqa_input_mib.png)

## Faster than Parquet with compression as good as its best

RowPack is a row-major dataset container for multimodal training workloads.
It is built for the pattern VLM training usually wants: sample a random window,
read a small block of neighboring rows, decode images, tokenize text, and feed
the batch without spending most of the step waiting on the loader.

Parquet is excellent for analytics, column scans, and ecosystem compatibility.
RowPack aims at a different hot path: training-time row/window access.

## Why It Is Useful

- Row-major layout enables speed by matching random-access shuffle training better than column-major
  Parquet files.
- Header-only C++ with Python bindings and PyTorch dataloader integration.
- Examples and writer APIs for recording new datasets or converting existing ones to RowPack.
- Modern libraries for speed and utility:
  - [nanobind](https://github.com/wjakob/nanobind) (BSD3) for fast Python bindings.
  - [CISTA](https://github.com/felixguendling/cista) (MIT) cast-mode payloads avoid rebuilding large Python dictionaries.
  - [LZAV](https://github.com/avaneev/lzav) (MIT) block compression gives good size reduction with very fast decompression.
  - [QOI](https://github.com/phoboslab/qoi) (MIT) for fast lossless image decoding and storage.
  - [STB](https://github.com/nothings/stb) (MIT) image decode and JPEG writing.
- The Python loader can hand back ready-to-shape byte buffers, so users can go
  straight to `np.frombuffer(...).reshape(h, w, c)`, or let the
  [PyTorch dataloader](torch_dataset.py) take care of batching.

In the current `mm_infographic_vqa` random-block benchmark, RowPack with LZAV
high-ratio blocks compresses close to Parquet GZIP/Brotli size while keeping
throughput near the uncompressed row-major baseline and well ahead of the
slower Parquet codecs.

## Portable and Self Contained

The native build is self-contained under `third_party/`, and the C++ writer is
header-only for easy embedding. The bundled dependencies and this repository
are permissively licensed.

## Build

From the repository root:

```bash
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

That builds the `rowpack_native` Python extension, builds the C++ example, and
runs the C++/Python smoke tests. Advanced layouts, including parent-directory
builds and explicit Python interpreter selection, are covered in
[docs/build-details.md](docs/build-details.md).

## Quick Examples

Create and read a tiny RowPack file with these verbose examples read further down for deeper explanation if interested.:

```bash
python3 examples/create_rowpack_direct.py --output build/examples/robot_demo.rowpack
python3 examples/read_rowpack.py --input build/examples/robot_demo.rowpack
```

Add LZAV compression and CISTA format for speed.

```bash
python3 examples/create_rowpack_direct.py \
  --output build/examples/robot_demo_native.rowpack \
  --payload-format cista \
  --block-codec lzav_hi \
  --native-module-dir build
```

Record dataset from ROS topics:

```bash
python3 -m rowpack.ros2_capture --config examples/capture_config.json
```

## Convert Parquet To RowPack

RowPack includes a generic Parquet converter. It streams Parquet batches,
turns each Parquet row into one RowPack row, and preserves ordinary columns as
JSON-compatible fields. Columns named `image`, `images`, `img`, or `imgs` are
treated as image payloads automatically; for other schemas, pass
`--image-column`.

```bash
python3 -m rowpack.convert_parquet \
  --input data/train.parquet \
  --output build/datasets/train.rowpack \
  --payload-format cista \
  --block-codec lzav_hi \
  --native-module-dir build \
  --image-column image \
  --name-column id \
  --overwrite
```

The converter requires `pyarrow`:

```bash
python3 -m pip install pyarrow
```

Useful options:

- `payload-format cista`: choose how each row is serialized inside RowPack.
  `json` is easy to inspect and useful for debugging. `cista` is the fast path:
  it stores typed native payloads that the C++ extension can read directly,
  avoiding a lot of Python object reconstruction during training.
- `image-column image`: move this Parquet column into RowPack's `images`
  payload list. Hugging Face-style image structs such as `{bytes, path}` work,
  as do raw encoded bytes. Repeat the option for multiple image columns.
- `block-codec lzav_hi`: choose block compression. Options are `none`,
  `lzav_default`, and `lzav_hi`. `none` is the pure layout baseline,
  `lzav_default` favors faster conversion, and `lzav_hi` spends more time while
  writing to get the smallest RowPack files. `lzav_hi` is the recommended
  default for dataset publishing; both LZAV modes are designed for very fast
  decompression during training.
- `rows-per-block 32`: compresses 32 rows into one block, a good default to
  balance compression ratio with shuffled random-window read performance.
- `name-column id`: use a stable Parquet column as the RowPack row name, so
  rows can be addressed by name later.
- `columns` and `drop-column`: limit which Parquet columns are read or stored.
  Non-image binary columns are preserved as base64 JSON wrappers so generic
  conversion does not silently throw bytes away.

The direct authoring example in [examples/create_rowpack_direct.py](examples/create_rowpack_direct.py)
uses the same writer API a Parquet converter would use.

For fully custom image handling, use `RowPackDatasetBuilder` directly. That is
where modes such as `raw_rgb`, `qoi_lossless`, and `jpeg_lossy` are available.

## Create RowPack Directly

RowPack can also be used as the dataset recording format. This is the path for
robots, simulators, data generation jobs, or any process that wants to append
rows online instead of converting from Parquet after the fact.

```bash
python3 examples/create_rowpack_direct.py --output build/examples/robot_demo.rowpack
```

The runnable source is [examples/create_rowpack_direct.py](examples/create_rowpack_direct.py).
`RowPackDatasetBuilder` defaults to `payload_format="cista"` and
`block_codec="lzav_hi"`. That means rows are serialized into compact native
payloads, then 32-row windows are compressed together by default. If you want a
debuggable file first, use `payload_format="json"` and `block_codec="none"`.

Image options:

- `encoded`: store source JPEG/PNG/WebP bytes as-is. Best when your source is
  already compressed.
- `raw_rgb`: store packed uint8 pixels with shape metadata.
- `qoi_lossless`: store raw RGB/RGBA pixels through QOI. Best for lossless
  source images when fast decode matters.
- `jpeg_lossy`: encode raw pixels through native STB JPEG writing. Best for
  camera streams where a small visual loss is acceptable.

## C++ Authoring

The C++ writer is header-only and mirrors the Python writer: rows accumulate in
a pending block, the block is optionally LZAV-compressed, and indexes plus
metadata are written when `finish()` is called.

The example is [examples/cpp_writer_smoke.cpp](examples/cpp_writer_smoke.cpp)
and is built by default:

```bash
./build/rowpack_cpp_writer_smoke --output build/examples/cpp_writer_smoke.rowpack
```

It writes one CISTA row with a JPEG image, reopens the file, and prints the
same kind of summary as the Python examples.

Use [include/rowpack/rowpack.hpp](include/rowpack/rowpack.hpp) for the reader/writer and
[include/rowpack/image_codecs.hpp](include/rowpack/image_codecs.hpp) when you want QOI or STB JPEG
helpers from C++. Define `ROWPACK_IMAGE_CODECS_IMPLEMENTATION` in exactly one
`.cpp` file that uses those codec helpers.

## ROS2 Capture

`rowpack.ros2_capture` records synchronized ROS2 topic groups into RowPack. It
does not require ROS2 to import RowPack; it only imports `rclpy` when the
capture command runs.

Edit [examples/capture_config.json](examples/capture_config.json) for your
topics, then run it on a ROS2 machine:

```bash
python3 -m rowpack.ros2_capture --config examples/capture_config.json
```

The first implementation assumes topic timestamps are already synchronized
within `sync.slop_s`. When all configured topics have arrived, it writes one
row containing JSON-compatible sensor values and any image payloads.

## Read From Python

Generic row reconstruction is shown in [examples/read_rowpack.py](examples/read_rowpack.py):

```bash
python3 examples/read_rowpack.py --input build/examples/robot_demo.rowpack
```

PyTorch list-file loader:

```python
from torch.utils.data import DataLoader
from rowpack import RowPackBlockDataset, RowPackLoaderState

dataset = RowPackBlockDataset(
    "data/variants/mm_infographic_vqa_rowpack/rowpacks.txt",
    mode="shuffle",
    return_format="native_vqa",
    state=RowPackLoaderState(file_index=0, block_index=0, seed=123),
)

loader = DataLoader(dataset, batch_size=16, num_workers=2, collate_fn=lambda batch: batch)

for batch in loader:
    row_id, text_pairs, images = batch[0]
    ...
```

The list file is plain text, one `.rowpack` path per line. In `sequential`
mode, `file_index` and `block_index` are the exact next list line and block. In
`shuffle` mode, those two values are deterministic counters mixed with `seed`
to choose a reproducible pseudo-random RowPack file and block. Rows are always
read sequentially inside a selected block.

Fast VQA iteration without `DataLoader`:

```python
from rowpack import RowPackBlockDataset

rows = RowPackBlockDataset(
    "data/variants/mm_infographic_vqa_rowpack/rowpacks.txt",
    mode="shuffle",
    return_format="native_vqa",
    seed=123,
)

for row_id, text_pairs, images in rows:
    image = images[0]
    # image["bytes"] is packed RGB when native STB/QOI decode succeeds.
    # Use np.frombuffer(image["bytes"], dtype=np.uint8).reshape(h, w, c).
```

## Benchmark

The benchmark scripts used to generate the published `mm_infographic_vqa`
charts are part of a sister repository used to demonstrate rowpack usage in a VLA,
and it will be published soon. 
However this repository includes a simplified RowPack-only benchmark as follows:

```bash
python3 examples/quick_benchmark.py \
  --payload-format cista \
  --block-codec lzav_hi \
  --native-module-dir build
```

That synthetic check writes and reads RowPack data only. It does not load
Parquet, convert Parquet, or compare against Parquet. It is useful for
confirming that the code and native module are working before you run a real
dataset comparison.

Example results on `nimapourjafar/mm_infographic_vqa`, using random-block
access, 32-row windows, 8,192 reproducibly sampled rows, 100 warmup steps, and
1,000 measured CPU training-loop steps:

| variant | size | samples/s | data wait |
| --- | ---: | ---: | ---: |
| parquet_uncompressed | 286.33 MiB | 22.99 | 27.65 ms |
| parquet_zstd | 262.27 MiB | 22.47 | 28.65 ms |
| parquet_gzip | 258.02 MiB | 14.28 | 54.10 ms |
| parquet_brotli | 258.37 MiB | 13.46 | 58.36 ms |
| rowpack_none | 287.59 MiB | 24.13 | 25.18 ms |
| rowpack_lzav_default | 266.04 MiB | 23.89 | 25.67 ms |
| rowpack_lzav_hi | 259.03 MiB | 23.88 | 25.65 ms |

![Mean data wait](docs/images/mm_infographic_vqa_data_wait_ms.png)

## Use With nanoVLM

Create a RowPack list file:

```text
data/variants/mm_infographic_vqa_rowpack/rowpack_cista_lzav_hi.rowpack
```

Then point `train.py` at that list:

```bash
python3 train.py \
  --rowpack_list data/variants/mm_infographic_vqa_rowpack/rowpacks.txt \
  --rowpack_read_mode shuffle \
  --rowpack_seed 123 \
  --batch_size 1 \
  --gradient_accumulation_steps 1 \
  --no_log_wandb
```

For a quick CPU-only integration check without downloading pretrained
backbones:

```bash
python3 train.py \
  --rowpack_list data/variants/mm_infographic_vqa_rowpack/rowpacks.txt \
  --rowpack_read_mode shuffle \
  --rowpack_seed 123 \
  --rowpack_max_rows 256 \
  --dataloader_num_workers 0 \
  --max_training_steps 1 \
  --batch_size 1 \
  --gradient_accumulation_steps 1 \
  --val_size 8 \
  --no_log_wandb \
  --no_eval \
  --no_lmms_eval \
  --tiny_debug_model
```
