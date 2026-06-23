#pragma once

#include "rowpack/rowpack.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#ifndef QOI_NO_STDIO
#define QOI_NO_STDIO
#endif

#ifndef STBI_WRITE_NO_STDIO
#define STBI_WRITE_NO_STDIO
#endif

#ifdef ROWPACK_IMAGE_CODECS_IMPLEMENTATION
#ifndef QOI_IMPLEMENTATION
#define QOI_IMPLEMENTATION
#endif
#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#endif

#include "qoi.h"
#include "stb_image_write.h"

namespace rowpack::image_codecs {

inline void require_image_shape(std::size_t bytes, std::uint32_t height, std::uint32_t width, std::uint32_t channels,
                                std::string_view codec) {
  require(height != 0 && width != 0, "Image encode requires nonzero width and height");
  require(channels == 1 || channels == 3 || channels == 4, "Image encode requires 1, 3, or 4 channels");
  auto const expected = static_cast<std::size_t>(height) * static_cast<std::size_t>(width) * channels;
  require(bytes == expected, std::string(codec) + " input byte length does not match height*width*channels");
}

inline std::vector<std::uint8_t> qoi_lossless(std::uint8_t const* rgb_or_rgba, std::uint32_t height,
                                             std::uint32_t width, std::uint32_t channels) {
  require(channels == 3 || channels == 4, "QOI requires RGB or RGBA input");
  require_image_shape(static_cast<std::size_t>(height) * width * channels, height, width, channels, "QOI");

  qoi_desc desc{};
  desc.width = width;
  desc.height = height;
  desc.channels = static_cast<unsigned char>(channels);
  desc.colorspace = QOI_SRGB;
  int out_len = 0;
  void* encoded = qoi_encode(rgb_or_rgba, &desc, &out_len);
  require(encoded != nullptr && out_len > 0, "qoi_encode failed");

  std::vector<std::uint8_t> out(static_cast<std::size_t>(out_len));
  std::memcpy(out.data(), encoded, out.size());
  std::free(encoded);
  return out;
}

inline std::vector<std::uint8_t> qoi_lossless(std::vector<std::uint8_t> const& rgb_or_rgba, std::uint32_t height,
                                             std::uint32_t width, std::uint32_t channels) {
  require_image_shape(rgb_or_rgba.size(), height, width, channels, "QOI");
  return qoi_lossless(rgb_or_rgba.data(), height, width, channels);
}

inline void stb_write_vector(void* context, void* data, int size) {
  auto* out = static_cast<std::vector<std::uint8_t>*>(context);
  auto const* bytes = static_cast<std::uint8_t const*>(data);
  out->insert(out->end(), bytes, bytes + size);
}

inline std::vector<std::uint8_t> jpeg_lossy(std::uint8_t const* pixels, std::size_t size, std::uint32_t height,
                                           std::uint32_t width, std::uint32_t channels, int quality = 90) {
  require_image_shape(size, height, width, channels, "JPEG");
  require(quality >= 1 && quality <= 100, "JPEG quality must be in [1, 100]");

  std::vector<std::uint8_t> out;
  auto ok = stbi_write_jpg_to_func(stb_write_vector, &out, static_cast<int>(width), static_cast<int>(height),
                                   static_cast<int>(channels), pixels, quality);
  require(ok != 0, "stbi_write_jpg_to_func failed");
  return out;
}

inline std::vector<std::uint8_t> jpeg_lossy(std::vector<std::uint8_t> const& pixels, std::uint32_t height,
                                           std::uint32_t width, std::uint32_t channels, int quality = 90) {
  return jpeg_lossy(pixels.data(), pixels.size(), height, width, channels, quality);
}

#ifdef ROWPACK_USE_CISTA
inline cast_payload::Image make_qoi_image(std::vector<std::uint8_t> const& rgb_or_rgba, std::uint32_t height,
                                          std::uint32_t width, std::uint32_t channels) {
  return make_image(qoi_lossless(rgb_or_rgba, height, width, channels), height, width, channels, "qoi_lossless");
}

inline cast_payload::Image make_jpeg_image(std::vector<std::uint8_t> const& pixels, std::uint32_t height,
                                           std::uint32_t width, std::uint32_t channels, int quality = 90) {
  return make_image(jpeg_lossy(pixels, height, width, channels, quality), height, width, channels, "encoded");
}
#endif

}  // namespace rowpack::image_codecs
