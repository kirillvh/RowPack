# Build Details

The README keeps the first-run path short. This page collects the CMake details
that are useful when RowPack is embedded in another project or when CMake picks
a different Python than the one you intended.

## Parent Or Downstream Builds

From a parent checkout, point `-S` at the RowPack source directory and choose any
build directory you like:

```bash
cmake -S rowpack -B rowpack_build
cmake --build rowpack_build --config Release
```

Downstream CMake projects can also add RowPack directly:

```cmake
add_subdirectory(path/to/rowpack)
target_link_libraries(your_target PRIVATE rowpack_headers)
```

## Python Selection

RowPack uses CMake's normal `FindPython` flow and asks it to prefer the active
virtual environment or PATH Python. The native module is specific to the Python
version it was built against. If CMake chooses the wrong interpreter, configure
with the interpreter you want:

```bash
cmake -S . -B build -DPython_EXECUTABLE=$(command -v python3)
cmake --build build --config Release
```

## Native Module Discovery

Python APIs that need `rowpack_native` look in these places, in order:

- `--native-module-dir`, when the API or example accepts it
- `ROWPACK_NATIVE_DIR`
- the normal Python import path
- common local CMake output directories such as `build/`, `build/Release/`,
  `rowpack_build/`, and `rowpack_build/Release/`

For unusual build layouts, set the environment variable explicitly:

```bash
export ROWPACK_NATIVE_DIR=/path/to/rowpack/build
```

## Rust Audio Helper

RowPack's Python audio API can use a bundled Rust helper for codec work:

- `flacenc-rs` writes standard FLAC streams for lossless audio payloads.
- `opus-rs` writes Opus packets into RowPack's small
  `rowpack_opus_packets` container. This is intentionally not an Ogg/Opus
  file; it is a training-friendly packet stream with sample rate, channels,
  frame size, sample count, and packet count carried in RowPack metadata.

Build it from the RowPack source directory:

```bash
cargo build --manifest-path tools/rowpack_audio_tool/Cargo.toml --release
```

Python APIs discover the helper in these places:

- `--audio-tool`, when an example accepts it
- `audio_tool=...`, when calling `encode_audio_payload`,
  `decode_audio_payload`, or `RowPackDatasetBuilder`
- `ROWPACK_AUDIO_TOOL`
- the normal `PATH`
- `tools/rowpack_audio_tool/target/release/`
- `tools/rowpack_audio_tool/target/debug/`

`backend="auto"` uses the Rust helper when it is found and falls back to
`ffmpeg` otherwise. `backend="rust"` requires the helper and raises a direct
error if it cannot be found. Arbitrary source files such as OGG/Vorbis still
use `ffmpeg` for the first normalization step to signed 16-bit PCM before the
Rust helper encodes FLAC or Opus.

## Native libavif Backend

RowPack ships a vendored `third_party/libavif` checkout, but native AVIF
encoding/decoding is opt-in because libavif still needs AV1 codec backends.
RowPack's default native AVIF build uses:

- `rav1e` for encoding, because it is a fast local AV1 encoder.
- `dav1d` for decoding, because it is a fast local AV1 decoder.
- no AOM, so this path does not require Perl.

The `rav1e` and `dav1d` source trees are vendored under
`third_party/libavif/ext/`, which lets libavif use them as local sources. You
still need the build tools those projects use:

- Rust/Cargo for `rav1e`.
- `cargo-c`, preferably the `cargo-cbuild` helper installed by
  `cargo install cargo-c --locked`. RowPack can also use `cargo-cinstall` when
  that helper works on the local machine.
- Meson and Ninja for `dav1d`.

Build with:

```bash
cmake -S . -B build-avif -DROWPACK_ENABLE_LIBAVIF=ON
cmake --build build-avif --config Release
```

That expands to libavif settings equivalent to:

```bash
-DAVIF_CODEC_AOM=OFF -DAVIF_CODEC_RAV1E=LOCAL -DAVIF_CODEC_DAV1D=LOCAL
```

### Windows Notes

The vendored dav1d integration is patched for RowPack portability:

- `AVIF_DAV1D_ENABLE_ASM=AUTO` is the default.
- If NASM or YASM is found, dav1d builds with assembly optimizations.
- If neither assembler is found, dav1d builds with `-Denable_asm=false` so the
  bundled build still works. Install NASM later if you want faster decode.

The vendored rav1e integration follows the same pattern:

- `AVIF_RAV1E_ENABLE_ASM=AUTO` is the default.
- If NASM or YASM is found, rav1e keeps its assembly optimizations.
- If neither assembler is found, rav1e builds without the `asm` feature so the
  bundled build still works. Install NASM later if you want faster encode.

If rav1e's cargo-c bootstrap hits a Windows file-copy, elevation, or permission
issue, install `cargo-c` once and point CMake at `cargo-cbuild`:

```bash
cargo install cargo-c --locked
cmake -S . -B build-avif ^
  -DROWPACK_ENABLE_LIBAVIF=ON ^
  -DCARGO_CBUILD=%USERPROFILE%\.cargo\bin\cargo-cbuild.exe
cmake --build build-avif --config Release
```

The patched rav1e integration also searches `%USERPROFILE%/.cargo/bin`,
`build-avif/`, and `build-avif/libavif/` for `cargo-cbuild` and
`cargo-cinstall`, so a successful bootstrap can be reused by later configures.

If your system already provides libavif and fast codec backends, use system
dependencies instead:

```bash
cmake -S . -B build-avif \
  -DROWPACK_ENABLE_LIBAVIF=ON \
  -DAVIF_CODEC_RAV1E=SYSTEM \
  -DAVIF_CODEC_DAV1D=SYSTEM
cmake --build build-avif --config Release
```

On Linux distributions that ship `libaom-dev` + `libdav1d-dev` but no
`librav1e-dev` package (e.g. Ubuntu 24.04), substitute AOM for the encoder:

```bash
cmake -S . -B build-avif \
  -DROWPACK_ENABLE_LIBAVIF=ON \
  -DAVIF_CODEC_AOM=SYSTEM \
  -DAVIF_CODEC_DAV1D=SYSTEM \
  -DAVIF_CODEC_RAV1E=OFF
cmake --build build-avif --config Release
```

AOM remains available if you explicitly choose it, but that is no longer the
RowPack default. On Windows, AOM's local build requires Perl. Optimized x86 AOM
builds also require NASM or YASM unless `-DAOM_TARGET_CPU=generic` is used.

After building, point examples at that native module if it is not in a normal
search path:

```bash
python examples/create_webcam_rowpack.py --encoder libavif --native-module-dir build-avif/Release
```

The regular `ffmpeg` encoder remains available and is still useful for H.264,
H.265, or machines where building AV1 codec dependencies is not convenient.
For AVIF, `encoder=auto` uses native libavif and reports a clear error if the
native module was not built or cannot be found. Pass
`--allow-ffmpeg-avif-fallback` only when you intentionally want to test the
system ffmpeg AVIF path.
