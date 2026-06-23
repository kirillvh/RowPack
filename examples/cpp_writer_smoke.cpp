#define ROWPACK_IMAGE_CODECS_IMPLEMENTATION

#include <rowpack/image_codecs.hpp>

#include <cstdint>
#include <vector>

int main() {
#ifndef ROWPACK_USE_CISTA
#error "This example needs CISTA support enabled."
#endif

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

  rowpack::Writer writer{"robot_capture.rowpack", options};

  rowpack::cast_payload::Row row;
  row.row_id = 0;
  row.extra_json = "{\"timestamp_ns\":123456789}";
  row.data.push_back(rowpack::make_turn("sensor", "text", "front camera frame"));

  std::vector<std::uint8_t> rgb = {
      255, 0, 0, 0, 255, 0,
      0, 0, 255, 255, 255, 255,
  };
  row.images.push_back(rowpack::image_codecs::make_jpeg_image(rgb, 2, 2, 3, 90));

  writer.append_cista_row(row, "frame_0", {"old_frame_0"});
  writer.finish();
  return 0;
}
