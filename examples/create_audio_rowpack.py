from __future__ import annotations

import argparse
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PARENT = SOURCE_ROOT.parent
if str(SOURCE_PARENT) not in sys.path:
    sys.path.insert(0, str(SOURCE_PARENT))

from rowpack import (
    MetadataBuilder,
    RowPackDatasetBuilder,
    RowPackReader,
    decode_audio_payload,
    probe_audio_path,
    rust_audio_tool_available,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a RowPack with original, FLAC, and Opus audio payloads.")
    parser.add_argument("--input", default="examples/sample_Into_the_Oceans_and_the_Air.ogg")
    parser.add_argument("--output", default="build/examples/audio_demo.rowpack")
    parser.add_argument("--payload-format", default="json", choices=["json", "cista"])
    parser.add_argument("--block-codec", default="none", choices=["none", "lzav_default", "lzav_hi"])
    parser.add_argument("--rows-per-block", type=int, default=8)
    parser.add_argument("--native-module-dir", default=None)
    parser.add_argument("--backend", default="auto", choices=["auto", "ffmpeg", "rust"])
    parser.add_argument("--audio-tool", default=None, help="Path to rowpack_audio_tool; otherwise auto-detect it.")
    parser.add_argument("--opus-bitrate", default="64k")
    parser.add_argument("--flac-compression-level", type=int, default=5)
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--skip-decode-check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = resolve_input_path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_info = probe_audio_path(input_path)
    metadata = (
        MetadataBuilder()
        .dataset_name("rowpack_audio_demo")
        .description("Audio RowPack example with original, FLAC lossless, and Opus lossy payloads.")
        .row_field("files", "file[]", "Audio payloads stored as RowPack file attachments")
        .row_field("_rowpack_audio", "json", "Audio row metadata")
        .audio_codec(
            "mixed",
            backend=args.backend,
            opus_bitrate=args.opus_bitrate,
            flac_compression_level=args.flac_compression_level,
            rust_audio_tool_available=rust_audio_tool_available(args.audio_tool),
        )
        .compression(block_codec=args.block_codec, rows_per_block=args.rows_per_block)
        .extra("source_audio", {"path": str(input_path), **source_info})
    )

    variants = [
        ("original_encoded", "encoded"),
        ("flac_lossless", "flac_lossless"),
        ("opus_lossy", "opus_lossy"),
    ]

    with RowPackDatasetBuilder(
        output_path,
        metadata=metadata,
        rows_per_block=args.rows_per_block,
        payload_format=args.payload_format,
        block_codec=args.block_codec,
        audio_backend=args.backend,
        audio_tool=args.audio_tool,
        opus_bitrate=args.opus_bitrate,
        flac_compression_level=args.flac_compression_level,
        native_module_dir=args.native_module_dir,
        overwrite=args.overwrite,
    ) as dataset:
        for row_index, (variant_name, codec) in enumerate(variants):
            # Each row stores one version of the same clip. This keeps the file
            # easy to inspect: row 0 is the source, row 1 is FLAC, row 2 is Opus.
            dataset.append_audio_row(
                [input_path],
                codec=codec,
                sample_rate=args.sample_rate,
                channels=args.channels,
                extra={"variant": variant_name, "source_path": str(input_path)},
                name=variant_name,
            )

    print(f"wrote {output_path}")
    print(f"source: {input_path} ({input_path.stat().st_size / 1024:.1f} KiB)")
    if source_info:
        duration = source_info.get("duration_s")
        duration_text = f"{duration:.2f} s" if isinstance(duration, (float, int)) else "unknown duration"
        print(
            "source audio: "
            f"{source_info.get('codec_name')} "
            f"{source_info.get('sample_rate')} Hz "
            f"{source_info.get('channels')} ch "
            f"{duration_text}"
        )

    with RowPackReader(output_path, native_module_dir=args.native_module_dir) as reader:
        print("\nvariant            stored KiB  codec          sample rate  channels  decoded PCM")
        print("-----------------  ----------  -------------  -----------  --------  -----------")
        for index in range(len(reader)):
            row = reader.read_row(index)
            audio = row["files"][0]
            decoded_summary = "skipped"
            if not args.skip_decode_check:
                decoded = decode_audio_payload(
                    audio,
                    sample_rate=args.sample_rate,
                    channels=args.channels,
                    audio_tool=args.audio_tool,
                )
                decoded_summary = (
                    f"{len(decoded['bytes']) / 1024:.1f} KiB, "
                    f"{decoded['sample_rate']} Hz, {decoded['channels']} ch"
                )
            print(
                f"{row['variant']:<17}  "
                f"{len(audio['bytes']) / 1024:>10.1f}  "
                f"{str(audio.get('audio_codec') or audio.get('codec')):<13}  "
                f"{str(audio.get('sample_rate')):>11}  "
                f"{str(audio.get('channels')):>8}  "
                f"{decoded_summary}"
            )

    return 0


def resolve_input_path(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    source_relative = SOURCE_ROOT / value
    if source_relative.exists():
        return source_relative
    examples_relative = SOURCE_ROOT / "examples" / value
    if examples_relative.exists():
        return examples_relative
    return path


if __name__ == "__main__":
    raise SystemExit(main())
