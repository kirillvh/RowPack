from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


class AudioCodecError(RuntimeError):
    pass


@dataclass(frozen=True)
class AudioCodecSettings:
    codec: str = "encoded"
    backend: str = "auto"
    opus_bitrate: str = "64k"
    flac_compression_level: int = 5
    sample_rate: int | None = None
    channels: int | None = None
    audio_tool: str | None = None
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"


def ffmpeg_available(ffmpeg: str = "ffmpeg") -> bool:
    return shutil.which(ffmpeg) is not None


def ffprobe_available(ffprobe: str = "ffprobe") -> bool:
    return shutil.which(ffprobe) is not None


def rust_audio_tool_available(audio_tool: str | None = None) -> bool:
    return find_rust_audio_tool(audio_tool) is not None


def encode_audio_payload(
    audio: Any,
    *,
    codec: str = "encoded",
    backend: str = "auto",
    opus_bitrate: str = "64k",
    flac_compression_level: int = 5,
    sample_rate: int | None = None,
    channels: int | None = None,
    name: str | None = None,
    audio_tool: str | None = None,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    **metadata: Any,
) -> dict[str, Any]:
    """Return a RowPack file payload containing encoded audio.

    ``backend='ffmpeg'`` writes standard FLAC/Ogg-Opus files. ``backend='rust'``
    uses RowPack's bundled helper: FLAC is a standard FLAC stream, while Opus is
    stored as a compact RowPack packet stream because opus-rs is a codec API,
    not an Ogg muxer.
    """

    codec = normalize_audio_codec(codec)
    backend = normalize_backend(backend)

    with audio_input_path(audio) as input_path:
        source_info = probe_audio(input_path, ffprobe=ffprobe)
        if codec == "encoded":
            payload = raw_audio_payload(audio, name=name)
            payload.update(audio_metadata_from_probe(source_info))
            payload.setdefault("codec", source_info.get("codec_name") or "encoded")
            payload.setdefault("audio_codec", payload["codec"])
            payload.setdefault("mime_type", guess_audio_mime(payload.get("name"), payload["codec"]))
            payload["role"] = "audio"
            payload["encoded_by"] = "source"
            payload.update(clean_metadata(metadata))
            payload["size"] = len(payload["bytes"])
            return payload

        backend = select_audio_backend(backend, codec, audio_tool=audio_tool)
        if backend == "rust":
            encoded, encoded_info, output_suffix, mime_type = run_rust_encode(
                input_path,
                codec=codec,
                source_info=source_info,
                opus_bitrate=opus_bitrate,
                sample_rate=sample_rate,
                channels=channels,
                audio_tool=audio_tool,
                ffmpeg=ffmpeg,
            )
        else:
            output_suffix = audio_extension(codec)
            mime_type = audio_mime_type(codec)
            with tempfile.TemporaryDirectory(prefix="rowpack_audio_") as temp_dir:
                output_path = Path(temp_dir) / f"encoded{output_suffix}"
                run_ffmpeg_encode(
                    input_path,
                    output_path,
                    codec=codec,
                    opus_bitrate=opus_bitrate,
                    flac_compression_level=flac_compression_level,
                    sample_rate=sample_rate,
                    channels=channels,
                    ffmpeg=ffmpeg,
                )
                encoded = output_path.read_bytes()
                encoded_info = probe_audio(output_path, ffprobe=ffprobe)

    payload_name = name or f"audio{output_suffix}"
    payload = {
        "bytes": encoded,
        "path": None,
        "name": payload_name,
        "mime_type": mime_type,
        "role": "audio",
        "codec": codec,
        "audio_codec": codec,
        "encoded_by": backend,
        "source_codec": source_info.get("codec_name"),
        "size": len(encoded),
        **audio_metadata_from_probe(encoded_info),
        **audio_metadata_from_tool(encoded_info),
    }
    payload.update(clean_metadata(metadata))
    return payload


def decode_audio_payload(
    audio: Any,
    *,
    sample_rate: int | None = None,
    channels: int | None = None,
    sample_format: str = "s16le",
    backend: str = "auto",
    audio_tool: str | None = None,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> dict[str, Any]:
    """Decode a RowPack audio payload to interleaved PCM bytes."""

    backend = normalize_backend(backend)
    if sample_format != "s16le":
        raise ValueError("Only sample_format='s16le' is currently supported")

    payload = raw_audio_payload(audio)
    codec = normalize_audio_codec_for_decode(str(payload.get("audio_codec") or payload.get("codec") or "encoded"))
    container = str(payload.get("container") or "")
    if backend == "auto":
        if container == "rowpack_opus_packets" or (
            payload.get("encoded_by") == "rust" and codec in {"flac_lossless", "opus_lossy"}
        ):
            backend = "rust"
        else:
            backend = "ffmpeg"

    with audio_input_path(audio) as input_path:
        if backend == "rust":
            return run_rust_decode(
                input_path,
                payload=payload,
                codec=codec,
                sample_rate=sample_rate,
                channels=channels,
                audio_tool=audio_tool,
                ffmpeg=ffmpeg,
            )

        source_info = probe_audio(input_path, ffprobe=ffprobe)
        out_sample_rate = int(sample_rate or source_info.get("sample_rate") or 48000)
        out_channels = int(channels or source_info.get("channels") or 2)
        with tempfile.TemporaryDirectory(prefix="rowpack_audio_decode_") as temp_dir:
            output_path = Path(temp_dir) / "decoded.pcm"
            cmd = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(input_path),
                "-vn",
                "-ar",
                str(out_sample_rate),
                "-ac",
                str(out_channels),
                "-f",
                sample_format,
                "-acodec",
                f"pcm_{sample_format}",
                str(output_path),
            ]
            run_command(cmd, "audio decode")
            pcm = output_path.read_bytes()

    bytes_per_sample = 2
    sample_count = len(pcm) // max(1, out_channels * bytes_per_sample)
    return {
        "bytes": pcm,
        "codec": f"pcm_{sample_format}",
        "audio_codec": f"pcm_{sample_format}",
        "mime_type": "audio/x-pcm",
        "sample_rate": out_sample_rate,
        "channels": out_channels,
        "sample_format": sample_format,
        "sample_count": sample_count,
        "duration_s": sample_count / out_sample_rate if out_sample_rate else None,
        "decoded_by": backend,
    }


def probe_audio_path(path: str | Path, *, ffprobe: str = "ffprobe") -> dict[str, Any]:
    return probe_audio(Path(path), ffprobe=ffprobe)


def probe_audio(path: Path, *, ffprobe: str = "ffprobe") -> dict[str, Any]:
    if not ffprobe_available(ffprobe):
        return {}
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_format",
        "-show_streams",
        "-print_format",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return {}
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}
    streams = parsed.get("streams") or []
    stream = streams[0] if streams else {}
    fmt = parsed.get("format") or {}
    return {
        "codec_name": stream.get("codec_name"),
        "sample_rate": int(stream["sample_rate"]) if str(stream.get("sample_rate") or "").isdigit() else None,
        "channels": int(stream["channels"]) if str(stream.get("channels") or "").isdigit() else None,
        "channel_layout": stream.get("channel_layout"),
        "sample_format": stream.get("sample_fmt"),
        "duration_s": float_value(stream.get("duration")) or float_value(fmt.get("duration")),
        "bitrate_bps": int_value(stream.get("bit_rate")) or int_value(fmt.get("bit_rate")),
        "container": fmt.get("format_name"),
    }


def select_audio_backend(backend: str, codec: str, *, audio_tool: str | None = None) -> str:
    if backend == "auto":
        if codec in {"flac_lossless", "opus_lossy"} and rust_audio_tool_available(audio_tool):
            return "rust"
        return "ffmpeg"
    if backend == "rust" and not rust_audio_tool_available(audio_tool):
        raise AudioCodecError(
            "RowPack's Rust audio helper was not found. Build it with "
            "`cargo build --manifest-path tools/rowpack_audio_tool/Cargo.toml --release`, "
            "pass audio_tool=..., or use backend='ffmpeg'."
        )
    return backend


def run_rust_encode(
    input_path: Path,
    *,
    codec: str,
    source_info: dict[str, Any],
    opus_bitrate: str,
    sample_rate: int | None,
    channels: int | None,
    audio_tool: str | None,
    ffmpeg: str,
) -> tuple[bytes, dict[str, Any], str, str]:
    tool = require_rust_audio_tool(audio_tool)
    if codec == "opus_lossy":
        target_sample_rate = int(sample_rate or 48000)
        output_suffix = ".rpopus"
        mime_type = "audio/x-rowpack-opus"
    elif codec == "flac_lossless":
        target_sample_rate = int(sample_rate or source_info.get("sample_rate") or 48000)
        output_suffix = ".flac"
        mime_type = "audio/flac"
    else:
        raise ValueError(f"Rust audio backend does not support {codec!r}")
    target_channels = int(channels or source_info.get("channels") or 2)

    with tempfile.TemporaryDirectory(prefix="rowpack_audio_rust_") as temp_dir:
        temp_path = Path(temp_dir)
        pcm_path = temp_path / "input.pcm"
        output_path = temp_path / f"encoded{output_suffix}"
        run_ffmpeg_decode_to_pcm(
            input_path,
            pcm_path,
            sample_rate=target_sample_rate,
            channels=target_channels,
            ffmpeg=ffmpeg,
        )
        if codec == "flac_lossless":
            command = [
                str(tool),
                "encode-flac",
                "--input-pcm",
                str(pcm_path),
                "--output",
                str(output_path),
                "--sample-rate",
                str(target_sample_rate),
                "--channels",
                str(target_channels),
            ]
        else:
            command = [
                str(tool),
                "encode-opus",
                "--input-pcm",
                str(pcm_path),
                "--output",
                str(output_path),
                "--sample-rate",
                str(target_sample_rate),
                "--channels",
                str(target_channels),
                "--bitrate-bps",
                str(parse_bitrate_bps(opus_bitrate)),
            ]
        info = run_rust_audio_command(command, f"audio encode {codec}")
        encoded = output_path.read_bytes()
    return encoded, info, output_suffix, mime_type


def run_rust_decode(
    input_path: Path,
    *,
    payload: dict[str, Any],
    codec: str,
    sample_rate: int | None,
    channels: int | None,
    audio_tool: str | None,
    ffmpeg: str,
) -> dict[str, Any]:
    tool = require_rust_audio_tool(audio_tool)
    container = payload.get("container")
    with tempfile.TemporaryDirectory(prefix="rowpack_audio_rust_decode_") as temp_dir:
        temp_path = Path(temp_dir)
        decoded_path = temp_path / "decoded.pcm"
        if codec == "flac_lossless":
            command = [
                str(tool),
                "decode-flac",
                "--input",
                str(input_path),
                "--output-pcm",
                str(decoded_path),
            ]
        elif codec == "opus_lossy" and container == "rowpack_opus_packets":
            command = [
                str(tool),
                "decode-opus",
                "--input",
                str(input_path),
                "--output-pcm",
                str(decoded_path),
            ]
        else:
            raise AudioCodecError(
                "Rust audio decode supports FLAC and RowPack Opus packet streams. "
                "Use backend='ffmpeg' for standard Ogg/Opus files."
            )
        info = run_rust_audio_command(command, f"audio decode {codec}")
        out_sample_rate = int(info.get("sample_rate") or payload.get("sample_rate") or 48000)
        out_channels = int(info.get("channels") or payload.get("channels") or 2)
        final_path = decoded_path
        requested_sample_rate = int(sample_rate or out_sample_rate)
        requested_channels = int(channels or out_channels)
        if requested_sample_rate != out_sample_rate or requested_channels != out_channels:
            converted_path = temp_path / "converted.pcm"
            convert_pcm_file(
                decoded_path,
                converted_path,
                in_sample_rate=out_sample_rate,
                in_channels=out_channels,
                out_sample_rate=requested_sample_rate,
                out_channels=requested_channels,
                ffmpeg=ffmpeg,
            )
            final_path = converted_path
            out_sample_rate = requested_sample_rate
            out_channels = requested_channels
        pcm = final_path.read_bytes()

    bytes_per_sample = 2
    sample_count = len(pcm) // max(1, out_channels * bytes_per_sample)
    return {
        "bytes": pcm,
        "codec": "pcm_s16le",
        "audio_codec": "pcm_s16le",
        "mime_type": "audio/x-pcm",
        "sample_rate": out_sample_rate,
        "channels": out_channels,
        "sample_format": "s16le",
        "sample_count": sample_count,
        "duration_s": sample_count / out_sample_rate if out_sample_rate else None,
        "decoded_by": "rust",
    }


def run_ffmpeg_encode(
    input_path: Path,
    output_path: Path,
    *,
    codec: str,
    opus_bitrate: str,
    flac_compression_level: int,
    sample_rate: int | None,
    channels: int | None,
    ffmpeg: str,
) -> None:
    if not ffmpeg_available(ffmpeg):
        raise AudioCodecError("ffmpeg was not found; install ffmpeg or build the Rust audio backend")
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(input_path), "-vn"]
    if sample_rate is not None:
        cmd.extend(["-ar", str(int(sample_rate))])
    if channels is not None:
        cmd.extend(["-ac", str(int(channels))])
    if codec == "flac_lossless":
        cmd.extend(["-c:a", "flac", "-compression_level", str(int(flac_compression_level))])
    elif codec == "opus_lossy":
        cmd.extend(["-c:a", "libopus", "-b:a", str(opus_bitrate), "-vbr", "on", "-application", "audio"])
    elif codec == "pcm_s16le":
        cmd.extend(["-f", "s16le", "-acodec", "pcm_s16le"])
    else:
        raise ValueError(f"Unsupported RowPack audio codec {codec!r}")
    cmd.append(str(output_path))
    run_command(cmd, f"audio encode {codec}")


def run_ffmpeg_decode_to_pcm(
    input_path: Path,
    output_path: Path,
    *,
    sample_rate: int,
    channels: int,
    ffmpeg: str,
) -> None:
    if not ffmpeg_available(ffmpeg):
        raise AudioCodecError("ffmpeg was not found; it is needed to normalize source audio to PCM")
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-ar",
        str(int(sample_rate)),
        "-ac",
        str(int(channels)),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    run_command(cmd, "audio PCM normalize")


def convert_pcm_file(
    input_path: Path,
    output_path: Path,
    *,
    in_sample_rate: int,
    in_channels: int,
    out_sample_rate: int,
    out_channels: int,
    ffmpeg: str,
) -> None:
    if not ffmpeg_available(ffmpeg):
        raise AudioCodecError("ffmpeg was not found; it is needed for PCM resampling/channel conversion")
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(int(in_sample_rate)),
        "-ac",
        str(int(in_channels)),
        "-i",
        str(input_path),
        "-ar",
        str(int(out_sample_rate)),
        "-ac",
        str(int(out_channels)),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    run_command(cmd, "audio PCM convert")


def run_command(cmd: list[str], label: str) -> None:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise AudioCodecError(f"ffmpeg {label} failed: {stderr}")


def run_rust_audio_command(cmd: list[str], label: str) -> dict[str, Any]:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise AudioCodecError(f"Rust {label} failed: {stderr}")
    return parse_key_value_output(result.stdout)


def parse_key_value_output(stdout: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key in {
            "sample_rate",
            "channels",
            "bits_per_sample",
            "sample_count",
            "bytes",
            "frame_ms",
            "frame_size",
            "packet_count",
            "bitrate_bps",
        }:
            parsed[key] = int(value)
        else:
            parsed[key] = value
    return parsed


@contextmanager
def audio_input_path(audio: Any) -> Iterator[Path]:
    if isinstance(audio, (str, Path)):
        yield Path(audio)
        return
    if isinstance(audio, dict) and audio.get("path") is not None and audio.get("bytes") is None:
        yield Path(audio["path"])
        return

    payload = raw_audio_payload(audio)
    suffix = audio_suffix_from_payload(payload)
    with tempfile.TemporaryDirectory(prefix="rowpack_audio_input_") as temp_dir:
        input_path = Path(temp_dir) / f"input{suffix}"
        input_path.write_bytes(payload["bytes"])
        yield input_path


def raw_audio_payload(audio: Any, *, name: str | None = None) -> dict[str, Any]:
    if isinstance(audio, dict):
        payload = dict(audio)
        if payload.get("bytes") is not None:
            payload["bytes"] = coerce_audio_bytes(payload["bytes"])
        elif payload.get("path") is not None:
            payload["bytes"] = Path(payload["path"]).read_bytes()
        else:
            raise TypeError("Audio payload dict requires 'bytes' or 'path'")
        if name is not None:
            payload["name"] = name
        payload.setdefault("path", None)
        payload.setdefault("name", Path(payload["path"]).name if payload.get("path") else name)
        payload.setdefault("mime_type", guess_audio_mime(payload.get("name"), payload.get("codec")))
        payload.setdefault("role", "audio")
        payload["size"] = len(payload["bytes"])
        return payload
    if isinstance(audio, (str, Path)):
        path = Path(audio)
        return {
            "bytes": path.read_bytes(),
            "path": str(path),
            "name": name or path.name,
            "mime_type": guess_audio_mime(name or path.name, None),
            "role": "audio",
            "size": path.stat().st_size,
        }
    data = coerce_audio_bytes(audio)
    return {
        "bytes": data,
        "path": None,
        "name": name,
        "mime_type": guess_audio_mime(name, None),
        "role": "audio",
        "size": len(data),
    }


def audio_metadata_from_probe(info: dict[str, Any]) -> dict[str, Any]:
    metadata = {}
    for key in ("sample_rate", "channels", "channel_layout", "sample_format", "duration_s", "bitrate_bps", "container"):
        value = info.get(key)
        if value is not None:
            metadata[key] = value
    return metadata


def audio_metadata_from_tool(info: dict[str, Any]) -> dict[str, Any]:
    metadata = {}
    for key in (
        "sample_rate",
        "channels",
        "bits_per_sample",
        "sample_count",
        "container",
        "frame_ms",
        "frame_size",
        "packet_count",
        "bitrate_bps",
    ):
        value = info.get(key)
        if value is not None:
            metadata[key] = value
    sample_rate = metadata.get("sample_rate")
    sample_count = metadata.get("sample_count")
    if sample_rate and sample_count and "duration_s" not in metadata:
        metadata["duration_s"] = float(sample_count) / float(sample_rate)
    return metadata


def clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if value is not None}


def coerce_audio_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    raise TypeError(f"Unsupported audio byte payload: {type(value)!r}")


def normalize_audio_codec(codec: str) -> str:
    normalized = codec.lower()
    aliases = {
        "source": "encoded",
        "copy": "encoded",
        "flac": "flac_lossless",
        "lossless_flac": "flac_lossless",
        "opus": "opus_lossy",
        "lossy_opus": "opus_lossy",
        "raw_pcm_s16le": "pcm_s16le",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"encoded", "flac_lossless", "opus_lossy", "pcm_s16le"}:
        raise ValueError(f"Unsupported RowPack audio codec {codec!r}")
    return normalized


def normalize_audio_codec_for_decode(codec: str) -> str:
    try:
        return normalize_audio_codec(codec)
    except ValueError:
        return "encoded"


def normalize_backend(backend: str) -> str:
    normalized = backend.lower()
    if normalized not in {"auto", "ffmpeg", "rust"}:
        raise ValueError("audio backend must be 'auto', 'ffmpeg', or 'rust'")
    return normalized


def find_rust_audio_tool(audio_tool: str | None = None) -> Path | None:
    executable = "rowpack_audio_tool.exe" if os.name == "nt" else "rowpack_audio_tool"
    candidates: list[Path] = []
    if audio_tool:
        candidates.append(Path(audio_tool))
    env_tool = os.environ.get("ROWPACK_AUDIO_TOOL")
    if env_tool:
        candidates.append(Path(env_tool))
    which = shutil.which(executable)
    if which:
        candidates.append(Path(which))

    root = Path(__file__).resolve().parent
    crate_dir = root / "tools" / "rowpack_audio_tool"
    candidates.extend(
        [
            crate_dir / "target" / "release" / executable,
            crate_dir / "target" / "debug" / executable,
        ]
    )
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def require_rust_audio_tool(audio_tool: str | None = None) -> Path:
    tool = find_rust_audio_tool(audio_tool)
    if tool is None:
        raise AudioCodecError(
            "RowPack's Rust audio helper was not found. Build it with "
            "`cargo build --manifest-path tools/rowpack_audio_tool/Cargo.toml --release`, "
            "set ROWPACK_AUDIO_TOOL, pass audio_tool=..., or use backend='ffmpeg'."
        )
    return tool


def parse_bitrate_bps(value: str) -> int:
    text = str(value).strip().lower()
    multiplier = 1
    if text.endswith("kbps"):
        multiplier = 1000
        text = text[:-4]
    elif text.endswith("k"):
        multiplier = 1000
        text = text[:-1]
    elif text.endswith("mbps"):
        multiplier = 1_000_000
        text = text[:-4]
    elif text.endswith("m"):
        multiplier = 1_000_000
        text = text[:-1]
    return int(float(text) * multiplier)


def audio_extension(codec: str) -> str:
    return {
        "encoded": ".bin",
        "flac_lossless": ".flac",
        "opus_lossy": ".opus",
        "pcm_s16le": ".pcm",
    }[codec]


def audio_mime_type(codec: str) -> str:
    return {
        "encoded": "application/octet-stream",
        "flac_lossless": "audio/flac",
        "opus_lossy": "audio/ogg",
        "pcm_s16le": "audio/x-pcm",
    }[codec]


def audio_suffix_from_payload(payload: dict[str, Any]) -> str:
    if payload.get("container") == "rowpack_opus_packets":
        return ".rpopus"
    name = payload.get("name") or payload.get("path")
    if name:
        suffix = Path(str(name)).suffix
        if suffix:
            return suffix
    codec = normalize_audio_codec_for_decode(str(payload.get("audio_codec") or payload.get("codec") or "encoded"))
    return audio_extension(codec)


def guess_audio_mime(name: Any, codec: Any) -> str:
    if codec:
        normalized = str(codec).lower()
        if normalized in {"flac", "flac_lossless"}:
            return "audio/flac"
        if normalized in {"opus", "opus_lossy"}:
            return "audio/ogg"
        if normalized in {"pcm_s16le", "raw_pcm_s16le"}:
            return "audio/x-pcm"
        if normalized == "rowpack_opus_packets":
            return "audio/x-rowpack-opus"
    suffix = Path(str(name)).suffix.lower() if name else ""
    if suffix == ".flac":
        return "audio/flac"
    if suffix in {".opus", ".ogg"}:
        return "audio/ogg"
    if suffix == ".rpopus":
        return "audio/x-rowpack-opus"
    if suffix == ".wav":
        return "audio/wav"
    return "application/octet-stream"


def float_value(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def int_value(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
