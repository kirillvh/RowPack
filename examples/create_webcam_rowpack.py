from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any


SOURCE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PARENT = SOURCE_ROOT.parent
if str(SOURCE_PARENT) not in sys.path:
    sys.path.insert(0, str(SOURCE_PARENT))

from rowpack import MetadataBuilder, RowPackDatasetBuilder, RowPackReader
from rowpack.video import (
    VideoChunkBuffer,
    is_avif_codec,
    libavif_available,
    native_avif_unavailable_message,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Record an OpenCV webcam stream into AVIF/MP4 RowPack chunks.")
    parser.add_argument("--output", default="build/examples/webcam_chunks.rowpack", help="Output .rowpack path")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--seconds", type=float, default=15.0, help="Capture duration. 0 means run until interrupted.")
    parser.add_argument("--chunk-seconds", type=float, default=5.0, help="Seconds of video per RowPack continuation row")
    parser.add_argument("--fps", type=float, default=None, help="Override capture FPS stored in chunk metadata")
    parser.add_argument("--width", type=int, default=0, help="Requested camera width")
    parser.add_argument("--height", type=int, default=0, help="Requested camera height")
    parser.add_argument("--codec", default="avif", choices=["avif", "h264", "h265"])
    parser.add_argument("--encoder", default="auto", choices=["auto", "libavif", "ffmpeg"])
    parser.add_argument("--crf", type=int, default=30, help="ffmpeg CRF, or libavif quality fallback as 100-crf")
    parser.add_argument("--quality", type=int, default=None, help="libavif quality in [1, 100]")
    parser.add_argument("--speed", type=int, default=6, help="libavif speed in [0, 10], larger is faster")
    parser.add_argument("--max-threads", type=int, default=1, help="libavif encoder thread count")
    parser.add_argument("--yuv-format", default="yuv420", choices=["yuv420", "yuv422", "yuv444"])
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable for encoder=ffmpeg")
    parser.add_argument(
        "--allow-ffmpeg-avif-fallback",
        action="store_true",
        help="Let encoder=auto use system ffmpeg for AVIF if native libavif is unavailable",
    )
    parser.add_argument("--payload-format", default="json", choices=["json", "cista"])
    parser.add_argument("--block-codec", default="none", choices=["none", "lzav_default", "lzav_hi"])
    parser.add_argument("--native-module-dir", default=None)
    parser.add_argument("--preview", action="store_true", help="Show a small OpenCV preview window")
    parser.add_argument("--max-chunks", type=int, default=0, help="Stop after writing this many chunks. 0 disables.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    native_module_dir = args.native_module_dir or default_native_module_dir()
    if (
        is_avif_codec(args.codec)
        and args.encoder in {"auto", "libavif"}
        and (args.encoder == "libavif" or not args.allow_ffmpeg_avif_fallback)
        and not libavif_available(native_module_dir)
    ):
        raise RuntimeError(native_avif_unavailable_message(native_module_dir))

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("This example needs OpenCV Python: `python -m pip install opencv-python`") from exc

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open camera index {args.camera}")
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    buffer = VideoChunkBuffer(
        stream="opencv_webcam",
        chunk_seconds=args.chunk_seconds,
        codec=args.codec,
        encoder=args.encoder,
        fps=args.fps or camera_fps(cap, cv2),
        crf=args.crf,
        quality=args.quality,
        speed=args.speed,
        max_threads=args.max_threads,
        yuv_format=args.yuv_format,
        ffmpeg=args.ffmpeg,
        native_module_dir=native_module_dir,
        allow_ffmpeg_fallback=args.allow_ffmpeg_avif_fallback,
    )
    print(f"video encoder: {buffer.encoder_name}")

    metadata = (
        MetadataBuilder()
        .dataset_name("opencv_webcam_chunks")
        .description("OpenCV webcam capture stored as RowPack video continuation rows")
        .row_field("files", "file[]", "Encoded AVIF/MP4 chunks")
        .row_field("_rowpack_continuation", "json", "Video chunk continuation metadata")
        .sensor("opencv_webcam", "rgb8", topic=f"camera_index:{args.camera}", role="camera")
        .compression(block_codec=args.block_codec, rows_per_block=8)
        .extra(
            "webcam_capture",
            {
                "camera": args.camera,
                "chunk_seconds": args.chunk_seconds,
                "codec": args.codec,
                "encoder": args.encoder,
                "selected_encoder": buffer.encoder_name,
                "crf": args.crf,
                "quality": args.quality,
                "speed": args.speed,
                "max_threads": args.max_threads,
                "yuv_format": args.yuv_format,
            },
        )
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    chunk_count = 0
    started = time.perf_counter()
    try:
        with RowPackDatasetBuilder(
            output,
            metadata=metadata,
            payload_format=args.payload_format,
            block_codec=args.block_codec,
            native_module_dir=native_module_dir,
            overwrite=args.overwrite,
        ) as builder:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    print("OpenCV camera returned no frame; stopping capture")
                    break

                timestamp_ns = time.time_ns()
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                height, width = frame_rgb.shape[:2]
                chunk = buffer.add_frame(
                    timestamp_ns=timestamp_ns,
                    data=frame_rgb.tobytes(),
                    height=height,
                    width=width,
                    channels=3,
                )
                frame_count += 1

                if args.preview:
                    cv2.imshow("RowPack webcam capture", frame_bgr)
                    key = cv2.waitKey(1) & 0xFF
                    if key in {27, ord("q")}:
                        break

                if chunk is not None:
                    append_chunk(builder, chunk)
                    chunk_count += 1
                    print_chunk(chunk, output)
                    if args.max_chunks and chunk_count >= args.max_chunks:
                        break

                elapsed = time.perf_counter() - started
                if args.seconds > 0 and elapsed >= args.seconds:
                    break

            chunk = buffer.flush()
            if chunk is not None and (not args.max_chunks or chunk_count < args.max_chunks):
                append_chunk(builder, chunk)
                chunk_count += 1
                print_chunk(chunk, output)
    finally:
        cap.release()
        if args.preview:
            cv2.destroyAllWindows()

    with RowPackReader(output, native_module_dir=native_module_dir) as reader:
        print(f"\nwrote {output}")
        print(f"  chunks: {chunk_count}")
        print(f"  source frames: {frame_count}")
        print(f"  rowpack rows: {len(reader)}")
        print(f"  file size: {output.stat().st_size} bytes")
    return 0


def default_native_module_dir() -> str | None:
    for candidate in [
        SOURCE_ROOT / "build" / "Release",
        SOURCE_ROOT / "build",
        SOURCE_PARENT / "rowpack_build" / "Release",
        SOURCE_PARENT / "rowpack_build",
    ]:
        if candidate.exists():
            return str(candidate)
    return None


def camera_fps(cap: Any, cv2: Any) -> float | None:
    value = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    return value if value > 0.0 else None


def append_chunk(builder: RowPackDatasetBuilder, chunk: dict[str, Any]) -> None:
    builder.append_video_chunk_row(
        stream=str(chunk.get("stream") or "opencv_webcam"),
        chunk=chunk,
        chunk_index=int(chunk.get("chunk_index") or 0),
        codec=str(chunk.get("codec") or "avif"),
        mime_type=str(chunk.get("mime_type") or "image/avif"),
        start_timestamp_ns=int(chunk.get("start_timestamp_ns") or 0),
        end_timestamp_ns=int(chunk.get("end_timestamp_ns") or 0),
        frame_count=int(chunk.get("frame_count") or 0),
        fps=float(chunk["fps"]) if chunk.get("fps") is not None else None,
        extra={"encoder": chunk.get("encoder"), "capture_backend": "opencv"},
    )


def print_chunk(chunk: dict[str, Any], output: Path) -> None:
    print(
        "chunk {index}: {frames} frames, {size} bytes, encoder={encoder}, wrote {output}".format(
            index=chunk.get("chunk_index"),
            frames=chunk.get("frame_count"),
            size=len(chunk.get("bytes") or b""),
            encoder=chunk.get("encoder"),
            output=output,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
