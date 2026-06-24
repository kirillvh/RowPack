#define ROWPACK_IMAGE_CODECS_IMPLEMENTATION

#include <rowpack/image_codecs.hpp>

#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::string format_bytes(std::uint64_t size) {
  auto value = static_cast<double>(size);
  char const* units[] = {"bytes", "KiB", "MiB", "GiB"};
  constexpr auto unit_count = std::size_t{4};
  std::size_t unit_index = 0;
  while (value >= 1024.0 && unit_index + 1 < unit_count) {
    value /= 1024.0;
    ++unit_index;
  }

  std::ostringstream out;
  if (unit_index == 0) {
    out << static_cast<std::uint64_t>(value) << ' ' << units[unit_index];
  } else {
    out << std::fixed << std::setprecision(2) << value << ' ' << units[unit_index];
  }
  return out.str();
}

std::string compression_summary(std::uint64_t compressed, std::uint64_t uncompressed) {
  if (uncompressed == 0) {
    return "n/a";
  }

  std::ostringstream out;
  out << format_bytes(compressed) << " stored from " << format_bytes(uncompressed) << " raw block payload (";
  if (compressed <= uncompressed) {
    auto const factor = compressed == 0 ? 0.0 : static_cast<double>(uncompressed) / static_cast<double>(compressed);
    auto const savings = (1.0 - (static_cast<double>(compressed) / static_cast<double>(uncompressed))) * 100.0;
    out << std::fixed << std::setprecision(2) << factor << "x smaller, " << std::setprecision(1) << savings
        << "% savings)";
  } else {
    auto const factor = static_cast<double>(compressed) / static_cast<double>(uncompressed);
    auto const overhead = (factor - 1.0) * 100.0;
    out << std::fixed << std::setprecision(2) << factor << "x larger, " << std::setprecision(1) << overhead
        << "% overhead)";
  }
  return out.str();
}

}  // namespace

int main(int argc, char** argv) {
#ifndef ROWPACK_USE_CISTA
#error "This example needs CISTA support enabled."
#endif

  // The executable can be run by CTest with no arguments, or by a human with an
  // explicit output path. Either way it creates one small, valid RowPack file.
  auto output_path = std::filesystem::path{"robot_capture.rowpack"};
  for (int arg_index = 1; arg_index < argc; ++arg_index) {
    auto const arg = std::string_view{argv[arg_index]};
    if (arg == "--output" && arg_index + 1 < argc) {
      output_path = argv[++arg_index];
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "usage: rowpack_cpp_writer_smoke [--output path/to/file.rowpack]\n";
      return 0;
    } else {
      output_path = argv[arg_index];
    }
  }

  auto metadata = rowpack::MetadataBuilder{}
                      .dataset_name("robot_smoke_capture")
                      .description("Minimal C++ RowPack authoring example")
                      .add_row_field("timestamp_ns", "uint64", "Sensor synchronization timestamp")
                      .add_sensor("front_camera", "rgb8", "Forward RGB camera", "{\"topic\":\"/camera/front\"}")
                      .set_compression_settings("lzav_hi", 32)
                      .set_image_codec_settings("jpeg_lossy", "{\"quality\":90}");

  rowpack::WriterOptions options;
  options.rows_per_block = 32;
  options.block_codec = rowpack::codec_id("lzav_hi");
  options.payload_format = "cista";
  options.metadata_json = metadata.to_json();
  options.overwrite = true;

  // The writer buffers rows until a block is full. On finish(), it compresses
  // any remaining rows, writes metadata, and writes the block/row indexes.
  rowpack::Writer writer{output_path, options};

  rowpack::cast_payload::Row row;
  row.row_id = 0;
  row.extra_json = "{\"timestamp_ns\":123456789}";
  row.data.push_back(rowpack::make_turn("sensor", "text", "front camera frame"));

  std::vector<std::uint8_t> rgb = {
      255, 0, 0, 0, 255, 0,
      0, 0, 255, 255, 255, 255,
  };
  // Store the tiny RGB image as JPEG bytes. For lossless experiments, the QOI
  // helpers in image_codecs.hpp follow the same pattern.
  row.images.push_back(rowpack::image_codecs::make_jpeg_image(rgb, 2, 2, 3, 90));

  writer.append_cista_row(row, "frame_0", {"old_frame_0"});
  writer.finish();

  // Reopen the file with the normal reader to prove the example wrote something
  // that training or conversion tools can consume.
  rowpack::Reader reader{output_path};
  auto compressed = std::uint64_t{0};
  auto uncompressed = std::uint64_t{0};
  for (auto const& block : reader.blocks()) {
    compressed += block.size;
    uncompressed += block.uncompressed_size;
  }
  auto const first_row_bytes = reader.read_row_bytes(0).size();

  std::cout << "\nCreated RowPack dataset\n";
  std::cout << "  path: " << output_path.string() << '\n';
  std::cout << "  file size: " << format_bytes(std::filesystem::file_size(output_path)) << '\n';
  std::cout << "  rows: " << reader.row_count() << '\n';
  std::cout << "  blocks: " << reader.block_count() << '\n';
  std::cout << "  payload format: cista\n";
  std::cout << "  block codec: lzav_hi\n";
  std::cout << "  block payload: " << compression_summary(compressed, uncompressed) << '\n';
  std::cout << "  row names: frame_0 (alias: old_frame_0)\n";
  std::cout << "  first row payload: " << format_bytes(first_row_bytes) << '\n';
  std::cout << "  first image: 2x2x3 RGB encoded as JPEG, " << format_bytes(row.images.front().bytes.size()) << '\n';
  std::cout << "  metadata stored:\n";
  std::cout << "    dataset_name: robot_smoke_capture\n";
  std::cout << "    row_schema: timestamp_ns uint64\n";
  std::cout << "    sensor: front_camera rgb8 topic=/camera/front\n";
  std::cout << "    block_payload_bytes: " << compressed << '\n';
  std::cout << "    block_uncompressed_bytes: " << uncompressed << '\n';
  std::cout << "    block_compression_ratio: "
            << (uncompressed == 0 ? 0.0 : static_cast<double>(compressed) / static_cast<double>(uncompressed)) << '\n';
  return 0;
}
