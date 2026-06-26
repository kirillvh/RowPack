#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>

#include "rowpack/rowpack.hpp"

#ifdef ROWPACK_USE_QOI
#define QOI_NO_STDIO
#define QOI_IMPLEMENTATION
#include "qoi.h"
#endif

#ifdef ROWPACK_USE_STB
#define STBI_NO_STDIO
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STBI_WRITE_NO_STDIO
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#endif

#ifdef ROWPACK_USE_LIBAVIF
#include "avif/avif.h"
#endif

namespace nb = nanobind;

namespace {

std::string bytes_string(nb::handle object) {
  char* buffer = nullptr;
  Py_ssize_t size = 0;
  if (PyBytes_AsStringAndSize(object.ptr(), &buffer, &size) != 0) {
    throw nb::python_error();
  }
  return {buffer, static_cast<std::size_t>(size)};
}

#ifdef ROWPACK_USE_LZAV
nb::bytes lzav_compress_bytes(nb::bytes payload, std::string const& codec) {
  auto const bytes = bytes_string(payload);
  if (bytes.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("LZAV input block exceeds supported size");
  }

  auto const input_size = static_cast<int>(bytes.size());
  int bound = 0;
  if (codec == "lzav_default") {
    bound = lzav_compress_bound(input_size);
  } else if (codec == "lzav_hi") {
    bound = lzav_compress_bound_hi(input_size);
  } else {
    throw std::runtime_error("Unknown LZAV codec: " + codec);
  }

  std::vector<std::uint8_t> out(static_cast<std::size_t>(bound));
  int compressed = 0;
  if (codec == "lzav_default") {
    compressed = lzav_compress_default(bytes.data(), out.data(), input_size, bound);
  } else {
    compressed = lzav_compress_hi(bytes.data(), out.data(), input_size, bound);
  }
  if (compressed == 0 && input_size != 0) {
    throw std::runtime_error("LZAV compression failed");
  }
  return nb::bytes(reinterpret_cast<char const*>(out.data()), static_cast<std::size_t>(compressed));
}

nb::bytes lzav_decompress_bytes(nb::bytes payload, std::uint64_t uncompressed_size) {
  auto const bytes = bytes_string(payload);
  if (bytes.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      uncompressed_size > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("LZAV block exceeds supported size");
  }

  std::vector<std::uint8_t> out(static_cast<std::size_t>(uncompressed_size));
  int decoded = lzav_decompress(bytes.data(), out.data(), static_cast<int>(bytes.size()), static_cast<int>(out.size()));
  if (decoded != static_cast<int>(out.size())) {
    throw std::runtime_error("LZAV decompression failed");
  }
  return nb::bytes(reinterpret_cast<char const*>(out.data()), out.size());
}
#endif

#ifdef ROWPACK_USE_LIBAVIF
int avif_quality_to_quantizer(int quality) {
  if (quality < 1 || quality > 100) {
    throw std::runtime_error("AVIF quality must be in [1, 100]");
  }
  return static_cast<int>(std::lround((100 - quality) * 63.0 / 99.0));
}

avifPixelFormat avif_pixel_format_from_string(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  if (value == "yuv420" || value == "420") {
    return AVIF_PIXEL_FORMAT_YUV420;
  }
  if (value == "yuv422" || value == "422") {
    return AVIF_PIXEL_FORMAT_YUV422;
  }
  if (value == "yuv444" || value == "444") {
    return AVIF_PIXEL_FORMAT_YUV444;
  }
  throw std::runtime_error("Unsupported AVIF pixel format: " + value);
}

nb::dict avif_runtime_info() {
  nb::dict info;
  char codec_versions[256] = {};
  avifCodecVersions(codec_versions);
  info["version"] = avifVersion();
  info["codec_versions"] = codec_versions;
  return info;
}

nb::bytes avif_encode_rgb_sequence(nb::list frames, std::uint32_t height, std::uint32_t width, double fps, int quality,
                                   int speed, int max_threads, std::string const& yuv_format) {
  if (frames.size() == 0) {
    throw std::runtime_error("AVIF encode requires at least one frame");
  }
  if (height == 0 || width == 0) {
    throw std::runtime_error("AVIF encode requires nonzero width and height");
  }
  if (fps <= 0.0) {
    throw std::runtime_error("AVIF encode requires fps > 0");
  }
  if (speed < AVIF_SPEED_SLOWEST || speed > AVIF_SPEED_FASTEST) {
    throw std::runtime_error("AVIF speed must be in [0, 10]");
  }

  auto const expected_size = static_cast<std::size_t>(height) * static_cast<std::size_t>(width) * 3U;
  auto const pixel_format = avif_pixel_format_from_string(yuv_format);
  auto const quantizer = avif_quality_to_quantizer(quality);
  auto encoder = std::unique_ptr<avifEncoder, decltype(&avifEncoderDestroy)>(avifEncoderCreate(), avifEncoderDestroy);
  if (!encoder) {
    throw std::runtime_error("avifEncoderCreate failed");
  }
  encoder->maxThreads = max_threads <= 0 ? 1 : max_threads;
  encoder->speed = speed;
  encoder->minQuantizer = quantizer;
  encoder->maxQuantizer = quantizer;
  encoder->minQuantizerAlpha = quantizer;
  encoder->maxQuantizerAlpha = quantizer;
  encoder->timescale = 1'000'000;

  auto const duration = std::max<std::uint64_t>(1, static_cast<std::uint64_t>(std::llround(1'000'000.0 / fps)));
  auto const frame_count = frames.size();
  for (std::size_t index = 0; index < frame_count; ++index) {
    auto const bytes = bytes_string(frames[index]);
    if (bytes.size() != expected_size) {
      throw std::runtime_error("AVIF frame byte length does not match height*width*3");
    }

    auto image = std::unique_ptr<avifImage, decltype(&avifImageDestroy)>(
        avifImageCreate(width, height, 8, pixel_format), avifImageDestroy);
    if (!image) {
      throw std::runtime_error("avifImageCreate failed");
    }
    image->yuvRange = AVIF_RANGE_FULL;

    avifRGBImage rgb;
    avifRGBImageSetDefaults(&rgb, image.get());
    rgb.format = AVIF_RGB_FORMAT_RGB;
    rgb.depth = 8;
    rgb.pixels = reinterpret_cast<std::uint8_t*>(const_cast<char*>(bytes.data()));
    rgb.rowBytes = static_cast<std::uint32_t>(width * 3U);

    auto result = avifImageRGBToYUV(image.get(), &rgb);
    if (result != AVIF_RESULT_OK) {
      throw std::runtime_error(std::string("avifImageRGBToYUV failed: ") + avifResultToString(result));
    }

    auto const flags = frame_count == 1 ? AVIF_ADD_IMAGE_FLAG_SINGLE : AVIF_ADD_IMAGE_FLAG_NONE;
    result = avifEncoderAddImage(encoder.get(), image.get(), duration, flags);
    if (result != AVIF_RESULT_OK) {
      throw std::runtime_error(std::string("avifEncoderAddImage failed: ") + avifResultToString(result));
    }
  }

  avifRWData output = AVIF_DATA_EMPTY;
  auto result = avifEncoderFinish(encoder.get(), &output);
  if (result != AVIF_RESULT_OK) {
    throw std::runtime_error(std::string("avifEncoderFinish failed: ") + avifResultToString(result));
  }
  nb::bytes encoded(reinterpret_cast<char const*>(output.data), output.size);
  avifRWDataFree(&output);
  return encoded;
}

nb::dict avif_decode_rgb_sequence(nb::bytes payload, int max_threads) {
  auto const bytes = bytes_string(payload);
  if (bytes.empty()) {
    throw std::runtime_error("AVIF decode requires non-empty bytes");
  }

  auto decoder = std::unique_ptr<avifDecoder, decltype(&avifDecoderDestroy)>(avifDecoderCreate(), avifDecoderDestroy);
  if (!decoder) {
    throw std::runtime_error("avifDecoderCreate failed");
  }
  decoder->maxThreads = max_threads <= 0 ? 1 : max_threads;

  auto result = avifDecoderSetIOMemory(
      decoder.get(),
      reinterpret_cast<std::uint8_t const*>(bytes.data()),
      bytes.size());
  if (result != AVIF_RESULT_OK) {
    throw std::runtime_error(std::string("avifDecoderSetIOMemory failed: ") + avifResultToString(result));
  }

  result = avifDecoderParse(decoder.get());
  if (result != AVIF_RESULT_OK) {
    throw std::runtime_error(std::string("avifDecoderParse failed: ") + avifResultToString(result));
  }

  nb::list frames;
  while (true) {
    result = avifDecoderNextImage(decoder.get());
    if (result == AVIF_RESULT_NO_IMAGES_REMAINING) {
      break;
    }
    if (result != AVIF_RESULT_OK) {
      throw std::runtime_error(std::string("avifDecoderNextImage failed: ") + avifResultToString(result));
    }

    avifRGBImage rgb;
    avifRGBImageSetDefaults(&rgb, decoder->image);
    rgb.format = AVIF_RGB_FORMAT_RGB;
    rgb.depth = 8;
    auto allocation = avifRGBImageAllocatePixels(&rgb);
    if (allocation != AVIF_RESULT_OK) {
      throw std::runtime_error(std::string("avifRGBImageAllocatePixels failed: ") + avifResultToString(allocation));
    }

    auto conversion = avifImageYUVToRGB(decoder->image, &rgb);
    if (conversion != AVIF_RESULT_OK) {
      avifRGBImageFreePixels(&rgb);
      throw std::runtime_error(std::string("avifImageYUVToRGB failed: ") + avifResultToString(conversion));
    }

    auto const frame_size = static_cast<std::size_t>(decoder->image->height) *
                            static_cast<std::size_t>(decoder->image->width) * 3U;
    frames.append(nb::bytes(reinterpret_cast<char const*>(rgb.pixels), frame_size));
    avifRGBImageFreePixels(&rgb);
  }

  nb::dict decoded;
  decoded["frames"] = frames;
  decoded["width"] = nb::int_(decoder->image ? decoder->image->width : 0);
  decoded["height"] = nb::int_(decoder->image ? decoder->image->height : 0);
  decoded["channels"] = nb::int_(3);
  decoded["frame_count"] = nb::int_(frames.size());
  decoded["timescale"] = nb::int_(decoder->timescale);
  decoded["duration_in_timescales"] = nb::int_(decoder->durationInTimescales);
  decoded["duration_s"] = nb::float_(decoder->timescale ? static_cast<double>(decoder->durationInTimescales) /
                                                           static_cast<double>(decoder->timescale)
                                                     : 0.0);
  decoded["fps"] = nb::float_(decoder->durationInTimescales && decoder->timescale
                                  ? static_cast<double>(frames.size()) * static_cast<double>(decoder->timescale) /
                                        static_cast<double>(decoder->durationInTimescales)
                                  : 0.0);
  return decoded;
}
#endif

#ifdef ROWPACK_USE_CISTA

std::string dict_string(nb::dict const& dict, char const* key) {
  if (!dict.contains(key)) {
    return {};
  }
  return nb::cast<std::string>(dict[key]);
}

bool is_bytes(nb::handle object) {
  return PyBytes_Check(object.ptr()) != 0;
}

std::uint32_t dict_uint32(nb::dict const& dict, char const* key) {
  if (!dict.contains(key)) {
    return 0;
  }
  return nb::cast<std::uint32_t>(dict[key]);
}

std::string dict_bytes(nb::dict const& dict, char const* key) {
  if (!dict.contains(key)) {
    return {};
  }
  return bytes_string(dict[key]);
}

#ifdef ROWPACK_USE_QOI
std::string qoi_encode_rgb(std::string const& raw, std::uint32_t height, std::uint32_t width, std::uint32_t channels) {
  if (height == 0 || width == 0 || (channels != 3 && channels != 4)) {
    throw std::runtime_error("QOI encode requires nonzero width/height and 3 or 4 channels");
  }
  auto const expected_size = static_cast<std::size_t>(height) * width * channels;
  if (raw.size() != expected_size) {
    throw std::runtime_error("QOI encode input byte length does not match height*width*channels");
  }

  qoi_desc desc{};
  desc.width = width;
  desc.height = height;
  desc.channels = static_cast<unsigned char>(channels);
  desc.colorspace = QOI_SRGB;
  int out_len = 0;
  void* encoded = qoi_encode(raw.data(), &desc, &out_len);
  if (encoded == nullptr || out_len <= 0) {
    throw std::runtime_error("qoi_encode failed");
  }

  std::string out(static_cast<char const*>(encoded), static_cast<std::size_t>(out_len));
  std::free(encoded);
  return out;
}

nb::dict qoi_decode_to_raw_dict(std::uint8_t const* data, std::size_t size) {
  qoi_desc desc{};
  void* decoded = qoi_decode(data, static_cast<int>(size), &desc, 3);
  if (decoded == nullptr) {
    throw std::runtime_error("qoi_decode failed");
  }

  auto const decoded_size = static_cast<std::size_t>(desc.height) * desc.width * 3U;
  nb::dict image_dict;
  image_dict["bytes"] = nb::bytes(static_cast<char const*>(decoded), decoded_size);
  image_dict["height"] = nb::int_(desc.height);
  image_dict["width"] = nb::int_(desc.width);
  image_dict["channels"] = nb::int_(3);
  image_dict["storage"] = "raw_rgb";
  std::free(decoded);
  return image_dict;
}
#endif

nb::dict raw_rgb_dict(char const* data, std::size_t size, std::uint32_t height, std::uint32_t width) {
  nb::dict image_dict;
  image_dict["bytes"] = nb::bytes(data, size);
  image_dict["height"] = nb::int_(height);
  image_dict["width"] = nb::int_(width);
  image_dict["channels"] = nb::int_(3);
  image_dict["storage"] = "raw_rgb";
  return image_dict;
}

#ifdef ROWPACK_USE_STB
bool stb_decode_to_raw_dict(std::uint8_t const* data, std::size_t size, nb::dict& out) {
  int width = 0;
  int height = 0;
  int channels_in_file = 0;
  auto* decoded = stbi_load_from_memory(
      reinterpret_cast<stbi_uc const*>(data),
      static_cast<int>(size),
      &width,
      &height,
      &channels_in_file,
      3);
  if (decoded == nullptr) {
    return false;
  }

  auto const decoded_size = static_cast<std::size_t>(height) * static_cast<std::size_t>(width) * 3U;
  out = raw_rgb_dict(reinterpret_cast<char const*>(decoded), decoded_size, static_cast<std::uint32_t>(height),
                     static_cast<std::uint32_t>(width));
  stbi_image_free(decoded);
  return true;
}

void stb_write_to_string(void* context, void* data, int size) {
  auto* out = static_cast<std::string*>(context);
  out->append(static_cast<char const*>(data), static_cast<std::size_t>(size));
}

nb::bytes jpeg_encode_rgb_bytes(nb::bytes payload, std::uint32_t height, std::uint32_t width, std::uint32_t channels,
                                int quality) {
  auto const bytes = bytes_string(payload);
  if (height == 0 || width == 0 || (channels != 1 && channels != 3 && channels != 4)) {
    throw std::runtime_error("JPEG encode requires nonzero width/height and 1, 3, or 4 channels");
  }
  auto const expected = static_cast<std::size_t>(height) * static_cast<std::size_t>(width) * channels;
  if (bytes.size() != expected) {
    throw std::runtime_error("JPEG encode input byte length does not match height*width*channels");
  }
  if (quality < 1 || quality > 100) {
    throw std::runtime_error("JPEG quality must be in [1, 100]");
  }

  std::string out;
  auto ok = stbi_write_jpg_to_func(stb_write_to_string, &out, static_cast<int>(width), static_cast<int>(height),
                                   static_cast<int>(channels), bytes.data(), quality);
  if (ok == 0) {
    throw std::runtime_error("stbi_write_jpg_to_func failed");
  }
  return nb::bytes(out.data(), out.size());
}
#endif

nb::dict stored_image_to_python_dict(rowpack::cast_payload::Image const& image) {
  auto const storage = image.storage.str();
  nb::dict image_dict;
  image_dict["bytes"] = nb::bytes(reinterpret_cast<char const*>(image.bytes.data()), image.bytes.size());
  image_dict["height"] = nb::int_(image.height);
  image_dict["width"] = nb::int_(image.width);
  image_dict["channels"] = nb::int_(image.channels);
  image_dict["storage"] = storage;
  return image_dict;
}

nb::dict direct_image_to_python_dict(rowpack::cast_payload::Image const& image, bool decode_images) {
  if (!decode_images) {
    return stored_image_to_python_dict(image);
  }

  auto const storage = image.storage.str();
#ifdef ROWPACK_USE_QOI
  if (storage == "qoi_lossless") {
    return qoi_decode_to_raw_dict(image.bytes.data(), image.bytes.size());
  }
#endif
#ifdef ROWPACK_USE_STB
  if (storage == "encoded") {
    nb::dict decoded;
    if (stb_decode_to_raw_dict(image.bytes.data(), image.bytes.size(), decoded)) {
      return decoded;
    }
  }
#endif
  return stored_image_to_python_dict(image);
}

std::string trim_copy(std::string value) {
  auto const is_not_space = [](unsigned char ch) { return !std::isspace(ch); };
  value.erase(value.begin(), std::find_if(value.begin(), value.end(), is_not_space));
  value.erase(std::find_if(value.rbegin(), value.rend(), is_not_space).base(), value.end());
  return value;
}

nb::tuple cista_row_to_vqa_tuple(rowpack::cast_payload::Row const& row, bool decode_images = true) {
  nb::list pairs;
  nb::list images;
  std::vector<std::string> pending_user;

  for (auto const& turn : row.data) {
    if (turn.modality.str() != "text") {
      continue;
    }

    auto const role = turn.role.str();
    auto text = trim_copy(turn.data.str());
    if (text.empty()) {
      continue;
    }

    if (role == "user") {
      pending_user.push_back(std::move(text));
    } else if (role == "assistant") {
      std::string user_text;
      for (std::size_t i = 0; i < pending_user.size(); ++i) {
        if (i != 0) {
          user_text += '\n';
        }
        user_text += pending_user[i];
      }
      user_text = trim_copy(std::move(user_text));
      if (!user_text.empty()) {
        pairs.append(nb::make_tuple(user_text, text));
      }
      pending_user.clear();
    }
  }

  for (auto const& image : row.images) {
    images.append(direct_image_to_python_dict(image, decode_images));
  }

  return nb::make_tuple(nb::int_(row.row_id), pairs, images);
}

nb::bytes encode_cista_payload(std::uint64_t row_id, nb::list turns, nb::list images, std::string const& extra_json) {
  rowpack::cast_payload::Row row;
  row.row_id = row_id;
  row.extra_json = extra_json;

  for (nb::handle item : turns) {
    auto const turn_dict = nb::cast<nb::dict>(item);
    rowpack::cast_payload::Turn turn;
    turn.role = dict_string(turn_dict, "role");
    turn.modality = dict_string(turn_dict, "modality");
    turn.data = dict_string(turn_dict, "data");
    row.data.push_back(std::move(turn));
  }

  for (nb::handle item : images) {
    std::string bytes;
    rowpack::cast_payload::Image image;
    if (is_bytes(item)) {
      bytes = bytes_string(item);
      image.storage = "encoded";
    } else {
      auto const image_dict = nb::cast<nb::dict>(item);
      bytes = dict_bytes(image_dict, "bytes");
      image.height = dict_uint32(image_dict, "height");
      image.width = dict_uint32(image_dict, "width");
      image.channels = dict_uint32(image_dict, "channels");
      auto storage = dict_string(image_dict, "storage");
      image.storage = storage.empty() ? "encoded" : storage;
#ifdef ROWPACK_USE_QOI
      if (storage == "qoi_lossless") {
        bytes = qoi_encode_rgb(bytes, image.height, image.width, image.channels);
        image.channels = 3;
      }
#endif
    }
    image.bytes.resize(static_cast<std::uint32_t>(bytes.size()));
    std::copy(bytes.begin(), bytes.end(), image.bytes.begin());
    row.images.push_back(std::move(image));
  }

  auto encoded = cista::serialize(row);
  return nb::bytes(reinterpret_cast<char const*>(encoded.data()), encoded.size());
}

nb::tuple decode_cista_payload(nb::bytes payload) {
  auto const bytes = bytes_string(payload);
  auto const* row = cista::deserialize<rowpack::cast_payload::Row, cista::mode::CAST>(
      reinterpret_cast<std::uint8_t const*>(bytes.data()),
      reinterpret_cast<std::uint8_t const*>(bytes.data() + bytes.size()));

  nb::list turns;
  for (auto const& turn : row->data) {
    nb::dict turn_dict;
    turn_dict["role"] = turn.role.str();
    turn_dict["modality"] = turn.modality.str();
    turn_dict["data"] = turn.data.str();
    turns.append(turn_dict);
  }

  nb::list images;
  for (auto const& image : row->images) {
    images.append(stored_image_to_python_dict(image));
  }

  return nb::make_tuple(nb::int_(row->row_id), turns, images, row->extra_json.str());
}

nb::tuple decode_cista_vqa_payload(nb::bytes payload, bool decode_images = true) {
  auto const bytes = bytes_string(payload);
  auto const* row = cista::deserialize<rowpack::cast_payload::Row, cista::mode::CAST>(
      reinterpret_cast<std::uint8_t const*>(bytes.data()),
      reinterpret_cast<std::uint8_t const*>(bytes.data() + bytes.size()));
  return cista_row_to_vqa_tuple(*row, decode_images);
}

#endif

}  // namespace

NB_MODULE(rowpack_native, m) {
  nb::set_leak_warnings(false);

  nb::class_<rowpack::Reader>(m, "Reader")
      .def(nb::init<std::filesystem::path>())
      .def("row_count", &rowpack::Reader::row_count)
      .def("block_count", &rowpack::Reader::block_count)
      .def("metadata_json", &rowpack::Reader::metadata_json)
      .def("read_row_bytes", &rowpack::Reader::read_row_bytes)
      .def("read_window_bytes", &rowpack::Reader::read_window_bytes)
#ifdef ROWPACK_USE_CISTA
      .def("read_cista_vqa_row", [](rowpack::Reader& reader, std::uint64_t row, bool decode_images) {
        auto bytes = reader.read_row_bytes(row);
        auto const* decoded = cista::deserialize<rowpack::cast_payload::Row, cista::mode::CAST>(
            bytes.data(), bytes.data() + bytes.size());
        return cista_row_to_vqa_tuple(*decoded, decode_images);
      }, nb::arg("row"), nb::arg("decode_images") = true)
#endif
      ;

#ifdef ROWPACK_USE_CISTA
  m.def("encode_cista_payload", &encode_cista_payload);
  m.def("decode_cista_payload", &decode_cista_payload);
  m.def("decode_cista_vqa_payload", &decode_cista_vqa_payload, nb::arg("payload"), nb::arg("decode_images") = true);
#endif
#ifdef ROWPACK_USE_LZAV
  m.def("lzav_compress", &lzav_compress_bytes);
  m.def("lzav_decompress", &lzav_decompress_bytes);
#endif
#if defined(ROWPACK_USE_STB) && defined(ROWPACK_USE_CISTA)
  m.def("jpeg_encode_rgb", &jpeg_encode_rgb_bytes);
#endif
#ifdef ROWPACK_USE_QOI
  m.def(
      "qoi_encode_rgb",
      [](nb::bytes payload, std::uint32_t height, std::uint32_t width, std::uint32_t channels) {
        auto const raw = bytes_string(payload);
        auto encoded = qoi_encode_rgb(raw, height, width, channels);
        return nb::bytes(encoded.data(), encoded.size());
      },
      nb::arg("payload"), nb::arg("height"), nb::arg("width"), nb::arg("channels"));
  m.def(
      "qoi_decode_rgb",
      [](nb::bytes payload) {
        auto const raw = bytes_string(payload);
        return qoi_decode_to_raw_dict(
            reinterpret_cast<std::uint8_t const*>(raw.data()), raw.size());
      },
      nb::arg("payload"));
#endif
#ifdef ROWPACK_USE_LIBAVIF
  m.def("avif_runtime_info", &avif_runtime_info);
  m.def(
      "avif_encode_rgb_sequence",
      &avif_encode_rgb_sequence,
      nb::arg("frames"),
      nb::arg("height"),
      nb::arg("width"),
      nb::arg("fps"),
      nb::arg("quality") = 70,
      nb::arg("speed") = 6,
      nb::arg("max_threads") = 1,
      nb::arg("yuv_format") = "yuv420");
  m.def("avif_decode_rgb_sequence", &avif_decode_rgb_sequence, nb::arg("payload"), nb::arg("max_threads") = 1);
#endif
}
