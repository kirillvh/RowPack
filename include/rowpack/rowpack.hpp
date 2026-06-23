#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#ifdef ROWPACK_USE_CISTA
#include <cista/containers.h>
#include <cista/serialization.h>
#endif

#ifdef ROWPACK_USE_LZAV
#include "lzav.h"
#endif

namespace rowpack {

inline constexpr std::array<char, 8> kMagic = {'R', 'O', 'W', 'P', 'A', 'C', 'K', '\0'};
inline constexpr std::uint16_t kVersionMajor = 0;
inline constexpr std::uint16_t kVersionMinor = 1;
inline constexpr std::uint32_t kHeaderSize = 128;
inline constexpr std::uint64_t kFlagUncompressed = 1ULL << 0U;
inline constexpr std::uint32_t kCodecNone = 0;
inline constexpr std::uint32_t kCodecLzavDefault = 1;
inline constexpr std::uint32_t kCodecLzavHi = 2;

#ifdef ROWPACK_USE_CISTA
inline constexpr auto kCistaCastMode = cista::mode::CAST;

namespace cistadata = cista::offset;

namespace cast_payload {

struct Turn {
  cistadata::string role;
  cistadata::string modality;
  cistadata::string data;
};

struct Image {
  cistadata::vector<std::uint8_t> bytes;
  std::uint32_t height{};
  std::uint32_t width{};
  std::uint32_t channels{};
  cistadata::string storage;
};

struct Row {
  std::uint64_t row_id{};
  cistadata::string extra_json;
  cistadata::vector<Turn> data;
  cistadata::vector<Image> images;
};

}  // namespace cast_payload
#endif

#pragma pack(push, 1)
struct FileHeader {
  char magic[8];
  std::uint16_t major;
  std::uint16_t minor;
  std::uint32_t header_size;
  std::uint64_t flags;
  std::uint64_t row_count;
  std::uint64_t block_count;
  std::uint64_t data_offset;
  std::uint64_t metadata_offset;
  std::uint64_t metadata_size;
  std::uint64_t block_index_offset;
  std::uint64_t block_index_size;
  std::uint64_t row_index_offset;
  std::uint64_t row_index_size;
  char reserved[32];
};

struct RowIndexEntry {
  std::uint64_t offset;
  std::uint64_t size;
  std::uint32_t block_id;
  std::uint32_t row_in_block;
};

struct BlockIndexEntry {
  std::uint64_t start_row;
  std::uint64_t row_count;
  std::uint64_t offset;
  std::uint64_t size;
  std::uint64_t uncompressed_size;
  std::uint32_t codec;
  std::uint32_t reserved;
};
#pragma pack(pop)

static_assert(sizeof(FileHeader) == kHeaderSize);
static_assert(sizeof(RowIndexEntry) == 24);
static_assert(sizeof(BlockIndexEntry) == 48);
static_assert(std::is_trivially_copyable_v<FileHeader>);
static_assert(std::is_trivially_copyable_v<RowIndexEntry>);
static_assert(std::is_trivially_copyable_v<BlockIndexEntry>);

inline FileHeader make_empty_header() {
  FileHeader h{};
  std::memcpy(h.magic, kMagic.data(), kMagic.size());
  h.major = kVersionMajor;
  h.minor = kVersionMinor;
  h.header_size = kHeaderSize;
  h.flags = kFlagUncompressed;
  h.data_offset = kHeaderSize;
  return h;
}

inline void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

template <typename T>
inline void read_exact(std::ifstream& input, T& value) {
  input.read(reinterpret_cast<char*>(&value), sizeof(T));
  require(static_cast<bool>(input), "Unexpected end of RowPack file");
}

inline std::vector<std::uint8_t> read_bytes(std::ifstream& input, std::uint64_t offset, std::uint64_t size) {
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  input.seekg(static_cast<std::streamoff>(offset));
  input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  require(static_cast<bool>(input), "Failed to read RowPack bytes");
  return bytes;
}

class Reader {
 public:
  Reader() = default;
  explicit Reader(std::filesystem::path path) { open(std::move(path)); }

  void open(std::filesystem::path path) {
    path_ = std::move(path);
    input_.open(path_, std::ios::binary);
    require(input_.is_open(), "Failed to open RowPack file");

    read_exact(input_, header_);
    require(std::memcmp(header_.magic, kMagic.data(), kMagic.size()) == 0, "Bad RowPack magic");
    require(header_.major == kVersionMajor, "Unsupported RowPack major version");
    require(header_.header_size == kHeaderSize, "Unsupported RowPack header size");

    metadata_ = read_metadata();
    blocks_ = read_index<BlockIndexEntry>(header_.block_index_offset, header_.block_count);
    rows_ = read_index<RowIndexEntry>(header_.row_index_offset, header_.row_count);
  }

  [[nodiscard]] std::uint64_t row_count() const { return header_.row_count; }
  [[nodiscard]] std::uint64_t block_count() const { return header_.block_count; }
  [[nodiscard]] std::string const& metadata_json() const { return metadata_; }

  [[nodiscard]] std::vector<std::uint8_t> read_row_bytes(std::uint64_t row) {
    require(row < rows_.size(), "RowPack row index out of range");
    auto const& entry = rows_[static_cast<std::size_t>(row)];
    require(entry.block_id < blocks_.size(), "RowPack row references an invalid block");
    auto const& block = blocks_[entry.block_id];
    if (block.codec != kCodecNone) {
      auto const& block_bytes = read_uncompressed_block(entry.block_id);
      require(entry.offset + entry.size <= block_bytes.size(), "RowPack row slice exceeds block size");
      auto const begin = block_bytes.begin() + static_cast<std::ptrdiff_t>(entry.offset);
      auto const end = begin + static_cast<std::ptrdiff_t>(entry.size);
      return {begin, end};
    }
    return read_bytes(input_, entry.offset, entry.size);
  }

  [[nodiscard]] std::vector<std::vector<std::uint8_t>> read_window_bytes(std::uint64_t start, std::uint64_t count) {
    require(start <= rows_.size(), "RowPack window start out of range");
    auto const stop = std::min<std::uint64_t>(start + count, rows_.size());
    std::vector<std::vector<std::uint8_t>> out;
    out.reserve(static_cast<std::size_t>(stop - start));
    for (auto row = start; row < stop; ++row) {
      out.push_back(read_row_bytes(row));
    }
    return out;
  }

 private:
  template <typename T>
  std::vector<T> read_index(std::uint64_t offset, std::uint64_t count) {
    std::vector<T> index(static_cast<std::size_t>(count));
    if (count == 0) {
      return index;
    }
    input_.seekg(static_cast<std::streamoff>(offset));
    input_.read(reinterpret_cast<char*>(index.data()), static_cast<std::streamsize>(sizeof(T) * index.size()));
    require(static_cast<bool>(input_), "Failed to read RowPack index");
    return index;
  }

  std::string read_metadata() {
    auto bytes = read_bytes(input_, header_.metadata_offset, header_.metadata_size);
    return std::string(reinterpret_cast<char const*>(bytes.data()), bytes.size());
  }

  std::vector<std::uint8_t> const& read_uncompressed_block(std::uint32_t block_id) {
    if (cached_block_id_ == block_id) {
      return cached_block_;
    }

    auto const& block = blocks_[block_id];
    require(block.codec == kCodecLzavDefault || block.codec == kCodecLzavHi, "Unsupported RowPack block codec");
#ifdef ROWPACK_USE_LZAV
    require(block.size <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
            "LZAV block compressed size exceeds supported limit");
    require(block.uncompressed_size <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
            "LZAV block uncompressed size exceeds supported limit");

    auto compressed = read_bytes(input_, block.offset, block.size);
    cached_block_.assign(static_cast<std::size_t>(block.uncompressed_size), std::uint8_t{});
    int decoded = lzav_decompress(compressed.data(), cached_block_.data(), static_cast<int>(compressed.size()),
                                  static_cast<int>(cached_block_.size()));
    require(decoded == static_cast<int>(cached_block_.size()), "LZAV decompression failed");
    cached_block_id_ = block_id;
    return cached_block_;
#else
    throw std::runtime_error("This RowPack build does not include LZAV support");
#endif
  }

  std::filesystem::path path_;
  std::ifstream input_;
  FileHeader header_{};
  std::string metadata_;
  std::vector<BlockIndexEntry> blocks_;
  std::vector<RowIndexEntry> rows_;
  std::uint32_t cached_block_id_{std::numeric_limits<std::uint32_t>::max()};
  std::vector<std::uint8_t> cached_block_;
};

}  // namespace rowpack
