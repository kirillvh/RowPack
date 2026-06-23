#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <limits>
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

nb::dict direct_image_to_python_dict(rowpack::cast_payload::Image const& image) {
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

nb::tuple cista_row_to_vqa_tuple(rowpack::cast_payload::Row const& row) {
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
    images.append(direct_image_to_python_dict(image));
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

nb::tuple decode_cista_vqa_payload(nb::bytes payload) {
  auto const bytes = bytes_string(payload);
  auto const* row = cista::deserialize<rowpack::cast_payload::Row, cista::mode::CAST>(
      reinterpret_cast<std::uint8_t const*>(bytes.data()),
      reinterpret_cast<std::uint8_t const*>(bytes.data() + bytes.size()));
  return cista_row_to_vqa_tuple(*row);
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
      .def("read_cista_vqa_row", [](rowpack::Reader& reader, std::uint64_t row) {
        auto bytes = reader.read_row_bytes(row);
        auto const* decoded = cista::deserialize<rowpack::cast_payload::Row, cista::mode::CAST>(
            bytes.data(), bytes.data() + bytes.size());
        return cista_row_to_vqa_tuple(*decoded);
      })
#endif
      ;

#ifdef ROWPACK_USE_CISTA
  m.def("encode_cista_payload", &encode_cista_payload);
  m.def("decode_cista_payload", &decode_cista_payload);
  m.def("decode_cista_vqa_payload", &decode_cista_vqa_payload);
#endif
#ifdef ROWPACK_USE_LZAV
  m.def("lzav_compress", &lzav_compress_bytes);
  m.def("lzav_decompress", &lzav_decompress_bytes);
#endif
}
