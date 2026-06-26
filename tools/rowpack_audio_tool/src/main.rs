use flacenc::bitsink::ByteSink;
use flacenc::component::parser;
use flacenc::component::BitRepr;
use flacenc::component::Decode;
use flacenc::error::Verify;
use flacenc::source::MemSource;
use opus_rs::{Application, OpusDecoder, OpusEncoder};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

const OPUS_MAGIC: &[u8; 8] = b"RPAUDOP1";
const DEFAULT_FRAME_MS: u16 = 20;

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let command = args.next().ok_or_else(usage)?;
    let options = parse_options(args.collect())?;

    match command.as_str() {
        "encode-flac" => encode_flac(&options),
        "decode-flac" => decode_flac(&options),
        "encode-opus" => encode_opus(&options),
        "decode-opus" => decode_opus(&options),
        "help" | "--help" | "-h" => {
            println!("{}", usage());
            Ok(())
        }
        other => Err(format!("unknown command {other:?}\n\n{}", usage())),
    }
}

fn usage() -> String {
    [
        "RowPack audio helper",
        "",
        "Commands:",
        "  encode-flac --input-pcm in.pcm --output out.flac --sample-rate 48000 --channels 2",
        "  decode-flac --input in.flac --output-pcm out.pcm",
        "  encode-opus --input-pcm in.pcm --output out.rpopus --sample-rate 48000 --channels 2 --bitrate-bps 64000",
        "  decode-opus --input in.rpopus --output-pcm out.pcm",
    ]
    .join("\n")
}

fn parse_options(args: Vec<String>) -> Result<HashMap<String, String>, String> {
    let mut options = HashMap::new();
    let mut index = 0usize;
    while index < args.len() {
        let key = &args[index];
        if !key.starts_with("--") {
            return Err(format!("expected --option, got {key:?}"));
        }
        if index + 1 >= args.len() {
            return Err(format!("missing value for {key}"));
        }
        options.insert(key.trim_start_matches("--").to_string(), args[index + 1].clone());
        index += 2;
    }
    Ok(options)
}

fn required_path(options: &HashMap<String, String>, key: &str) -> Result<PathBuf, String> {
    options
        .get(key)
        .map(PathBuf::from)
        .ok_or_else(|| format!("missing --{key}"))
}

fn required_usize(options: &HashMap<String, String>, key: &str) -> Result<usize, String> {
    options
        .get(key)
        .ok_or_else(|| format!("missing --{key}"))?
        .parse::<usize>()
        .map_err(|err| format!("invalid --{key}: {err}"))
}

fn optional_u16(options: &HashMap<String, String>, key: &str, default: u16) -> Result<u16, String> {
    match options.get(key) {
        Some(value) => value.parse::<u16>().map_err(|err| format!("invalid --{key}: {err}")),
        None => Ok(default),
    }
}

fn optional_i32(options: &HashMap<String, String>, key: &str, default: i32) -> Result<i32, String> {
    match options.get(key) {
        Some(value) => value.parse::<i32>().map_err(|err| format!("invalid --{key}: {err}")),
        None => Ok(default),
    }
}

fn encode_flac(options: &HashMap<String, String>) -> Result<(), String> {
    let input = required_path(options, "input-pcm")?;
    let output = required_path(options, "output")?;
    let sample_rate = required_usize(options, "sample-rate")?;
    let channels = required_usize(options, "channels")?;
    let bits_per_sample = 16usize;
    let pcm = read_pcm_i16_as_i32(&input)?;

    if channels == 0 {
        return Err("channels must be positive".to_string());
    }
    let sample_count = pcm.len() / channels;
    let config = flacenc::config::Encoder::default()
        .into_verified()
        .map_err(|err| format!("FLAC config verification failed: {err:?}"))?;
    let source = MemSource::from_samples(&pcm, channels, bits_per_sample, sample_rate);
    let stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
        .map_err(|err| format!("FLAC encode failed: {err:?}"))?;
    let mut sink = ByteSink::new();
    stream
        .write(&mut sink)
        .map_err(|err| format!("FLAC bitstream write failed: {err:?}"))?;
    write_bytes(&output, sink.as_slice())?;

    print_kv(&[
        ("codec", "flac_lossless".to_string()),
        ("container", "flac".to_string()),
        ("sample_rate", sample_rate.to_string()),
        ("channels", channels.to_string()),
        ("bits_per_sample", bits_per_sample.to_string()),
        ("sample_count", sample_count.to_string()),
        ("bytes", sink.as_slice().len().to_string()),
    ]);
    Ok(())
}

fn decode_flac(options: &HashMap<String, String>) -> Result<(), String> {
    let input = required_path(options, "input")?;
    let output = required_path(options, "output-pcm")?;
    let bytes = fs::read(&input).map_err(|err| format!("failed to read {}: {err}", input.display()))?;
    let (_remaining, stream) = parser::stream::<nom::error::VerboseError<&[u8]>>(&bytes)
        .map_err(|err| format!("FLAC parse failed: {err:?}"))?;
    let info = stream.stream_info();
    let sample_rate = info.sample_rate();
    let channels = info.channels();
    let bits_per_sample = info.bits_per_sample();
    if bits_per_sample > 16 {
        return Err(format!(
            "only <=16-bit FLAC decode is currently written to s16le PCM, got {bits_per_sample}"
        ));
    }

    let mut pcm = Vec::<i16>::new();
    for frame_index in 0..stream.frame_count() {
        let frame = stream
            .frame(frame_index)
            .ok_or_else(|| format!("missing FLAC frame {frame_index}"))?;
        pcm.extend(frame.decode().into_iter().map(i32_to_i16));
    }
    write_pcm_i16(&output, &pcm)?;
    let sample_count = pcm.len() / channels.max(1);

    print_kv(&[
        ("codec", "pcm_s16le".to_string()),
        ("sample_rate", sample_rate.to_string()),
        ("channels", channels.to_string()),
        ("bits_per_sample", "16".to_string()),
        ("sample_count", sample_count.to_string()),
        ("bytes", (pcm.len() * 2).to_string()),
    ]);
    Ok(())
}

fn encode_opus(options: &HashMap<String, String>) -> Result<(), String> {
    let input = required_path(options, "input-pcm")?;
    let output = required_path(options, "output")?;
    let sample_rate = required_usize(options, "sample-rate")?;
    let channels = required_usize(options, "channels")?;
    let bitrate_bps = optional_i32(options, "bitrate-bps", 64000)?;
    let frame_ms = optional_u16(options, "frame-ms", DEFAULT_FRAME_MS)?;
    validate_opus_shape(sample_rate, channels, frame_ms)?;

    let pcm = read_pcm_i16(&input)?;
    let sample_count = pcm.len() / channels;
    let frame_size = sample_rate * frame_ms as usize / 1000;
    let mut encoder = OpusEncoder::new(sample_rate as i32, channels, Application::Audio)
        .map_err(|err| format!("Opus encoder init failed: {err}"))?;
    encoder.bitrate_bps = bitrate_bps;
    encoder.use_cbr = false;
    encoder.complexity = 5;

    let mut packets = Vec::<Vec<u8>>::new();
    let mut frame_start = 0usize;
    while frame_start < sample_count {
        let frames_to_copy = (sample_count - frame_start).min(frame_size);
        let mut frame = vec![0.0f32; frame_size * channels];
        for t in 0..frames_to_copy {
            for ch in 0..channels {
                let sample = pcm[(frame_start + t) * channels + ch];
                frame[t * channels + ch] = f32::from(sample) / 32768.0;
            }
        }

        let mut packet = vec![0u8; 4096];
        let len = encoder
            .encode(&frame, frame_size, &mut packet)
            .map_err(|err| format!("Opus encode failed: {err}"))?;
        if len > u16::MAX as usize {
            return Err(format!("Opus packet too large for RowPack packet stream: {len} bytes"));
        }
        packet.truncate(len);
        packets.push(packet);
        frame_start += frame_size;
    }

    let encoded = pack_opus_stream(sample_rate, channels, frame_ms, frame_size, sample_count, bitrate_bps, &packets)?;
    write_bytes(&output, &encoded)?;
    print_kv(&[
        ("codec", "opus_lossy".to_string()),
        ("container", "rowpack_opus_packets".to_string()),
        ("sample_rate", sample_rate.to_string()),
        ("channels", channels.to_string()),
        ("frame_ms", frame_ms.to_string()),
        ("frame_size", frame_size.to_string()),
        ("sample_count", sample_count.to_string()),
        ("packet_count", packets.len().to_string()),
        ("bitrate_bps", bitrate_bps.to_string()),
        ("bytes", encoded.len().to_string()),
    ]);
    Ok(())
}

fn decode_opus(options: &HashMap<String, String>) -> Result<(), String> {
    let input = required_path(options, "input")?;
    let output = required_path(options, "output-pcm")?;
    let bytes = fs::read(&input).map_err(|err| format!("failed to read {}: {err}", input.display()))?;
    let stream = unpack_opus_stream(&bytes)?;
    let mut decoder = OpusDecoder::new(stream.sample_rate as i32, stream.channels)
        .map_err(|err| format!("Opus decoder init failed: {err}"))?;
    let mut pcm = Vec::<i16>::with_capacity(stream.sample_count * stream.channels);
    let mut written_frames = 0usize;

    for packet in &stream.packets {
        let mut output_frame = vec![0.0f32; stream.frame_size * stream.channels];
        let decoded_frames = decoder
            .decode(packet, stream.frame_size, &mut output_frame)
            .map_err(|err| format!("Opus decode failed: {err}"))?;
        let remaining = stream.sample_count.saturating_sub(written_frames);
        let frames_to_write = decoded_frames.min(remaining);
        for sample in output_frame.iter().take(frames_to_write * stream.channels) {
            pcm.push(f32_to_i16(*sample));
        }
        written_frames += frames_to_write;
        if written_frames >= stream.sample_count {
            break;
        }
    }

    write_pcm_i16(&output, &pcm)?;
    print_kv(&[
        ("codec", "pcm_s16le".to_string()),
        ("sample_rate", stream.sample_rate.to_string()),
        ("channels", stream.channels.to_string()),
        ("sample_count", written_frames.to_string()),
        ("bytes", (pcm.len() * 2).to_string()),
    ]);
    Ok(())
}

fn validate_opus_shape(sample_rate: usize, channels: usize, frame_ms: u16) -> Result<(), String> {
    if !matches!(sample_rate, 8000 | 12000 | 16000 | 24000 | 48000) {
        return Err(format!(
            "opus-rs supports sample rates 8000, 12000, 16000, 24000, and 48000; got {sample_rate}"
        ));
    }
    if !matches!(channels, 1 | 2) {
        return Err(format!("opus-rs supports mono or stereo; got {channels} channels"));
    }
    if !matches!(frame_ms, 10 | 20 | 40 | 60) {
        return Err(format!("unsupported Opus frame duration {frame_ms} ms"));
    }
    Ok(())
}

fn pack_opus_stream(
    sample_rate: usize,
    channels: usize,
    frame_ms: u16,
    frame_size: usize,
    sample_count: usize,
    bitrate_bps: i32,
    packets: &[Vec<u8>],
) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    out.extend_from_slice(OPUS_MAGIC);
    push_u32(&mut out, sample_rate)?;
    push_u16(&mut out, channels)?;
    out.extend_from_slice(&frame_ms.to_le_bytes());
    push_u32(&mut out, frame_size)?;
    push_u64(&mut out, sample_count)?;
    push_u32(&mut out, packets.len())?;
    push_u32(&mut out, bitrate_bps.max(0) as usize)?;
    for packet in packets {
        push_u16(&mut out, packet.len())?;
        out.extend_from_slice(packet);
    }
    Ok(out)
}

struct OpusPacketStream {
    sample_rate: usize,
    channels: usize,
    frame_size: usize,
    sample_count: usize,
    packets: Vec<Vec<u8>>,
}

fn unpack_opus_stream(bytes: &[u8]) -> Result<OpusPacketStream, String> {
    if bytes.len() < 36 || &bytes[..8] != OPUS_MAGIC {
        return Err("not a RowPack Opus packet stream".to_string());
    }
    let mut cursor = 8usize;
    let sample_rate = take_u32(bytes, &mut cursor)? as usize;
    let channels = take_u16(bytes, &mut cursor)? as usize;
    let _frame_ms = take_u16(bytes, &mut cursor)?;
    let frame_size = take_u32(bytes, &mut cursor)? as usize;
    let sample_count = take_u64(bytes, &mut cursor)? as usize;
    let packet_count = take_u32(bytes, &mut cursor)? as usize;
    let _bitrate_bps = take_u32(bytes, &mut cursor)?;
    validate_opus_shape(sample_rate, channels, 20)?;

    let mut packets = Vec::with_capacity(packet_count);
    for _ in 0..packet_count {
        let len = take_u16(bytes, &mut cursor)? as usize;
        if cursor + len > bytes.len() {
            return Err("truncated RowPack Opus packet stream".to_string());
        }
        packets.push(bytes[cursor..cursor + len].to_vec());
        cursor += len;
    }
    Ok(OpusPacketStream {
        sample_rate,
        channels,
        frame_size,
        sample_count,
        packets,
    })
}

fn read_pcm_i16(path: &Path) -> Result<Vec<i16>, String> {
    let bytes = fs::read(path).map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    if bytes.len() % 2 != 0 {
        return Err(format!("PCM file {} has an odd byte count", path.display()));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn read_pcm_i16_as_i32(path: &Path) -> Result<Vec<i32>, String> {
    Ok(read_pcm_i16(path)?.into_iter().map(i32::from).collect())
}

fn write_pcm_i16(path: &Path, samples: &[i16]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    write_bytes(path, &bytes)
}

fn write_bytes(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    fs::write(path, bytes).map_err(|err| format!("failed to write {}: {err}", path.display()))
}

fn push_u16(out: &mut Vec<u8>, value: usize) -> Result<(), String> {
    let value = u16::try_from(value).map_err(|_| format!("value too large for u16: {value}"))?;
    out.write_all(&value.to_le_bytes()).map_err(|err| err.to_string())
}

fn push_u32(out: &mut Vec<u8>, value: usize) -> Result<(), String> {
    let value = u32::try_from(value).map_err(|_| format!("value too large for u32: {value}"))?;
    out.write_all(&value.to_le_bytes()).map_err(|err| err.to_string())
}

fn push_u64(out: &mut Vec<u8>, value: usize) -> Result<(), String> {
    let value = u64::try_from(value).map_err(|_| format!("value too large for u64: {value}"))?;
    out.write_all(&value.to_le_bytes()).map_err(|err| err.to_string())
}

fn take_u16(bytes: &[u8], cursor: &mut usize) -> Result<u16, String> {
    if *cursor + 2 > bytes.len() {
        return Err("truncated u16".to_string());
    }
    let value = u16::from_le_bytes([bytes[*cursor], bytes[*cursor + 1]]);
    *cursor += 2;
    Ok(value)
}

fn take_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, String> {
    if *cursor + 4 > bytes.len() {
        return Err("truncated u32".to_string());
    }
    let value = u32::from_le_bytes([bytes[*cursor], bytes[*cursor + 1], bytes[*cursor + 2], bytes[*cursor + 3]]);
    *cursor += 4;
    Ok(value)
}

fn take_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64, String> {
    if *cursor + 8 > bytes.len() {
        return Err("truncated u64".to_string());
    }
    let value = u64::from_le_bytes([
        bytes[*cursor],
        bytes[*cursor + 1],
        bytes[*cursor + 2],
        bytes[*cursor + 3],
        bytes[*cursor + 4],
        bytes[*cursor + 5],
        bytes[*cursor + 6],
        bytes[*cursor + 7],
    ]);
    *cursor += 8;
    Ok(value)
}

fn i32_to_i16(value: i32) -> i16 {
    value.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

fn f32_to_i16(value: f32) -> i16 {
    (value.clamp(-1.0, 1.0) * 32767.0).round() as i16
}

fn print_kv(values: &[(&str, String)]) {
    for (key, value) in values {
        println!("{key}={value}");
    }
}
