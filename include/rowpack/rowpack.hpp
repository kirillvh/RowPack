#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
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

inline std::string_view codec_name(std::uint32_t codec) {
  switch (codec) {
    case kCodecNone:
      return "none";
    case kCodecLzavDefault:
      return "lzav_default";
    case kCodecLzavHi:
      return "lzav_hi";
    default:
      return "unknown";
  }
}

inline std::uint32_t codec_id(std::string_view codec) {
  if (codec == "none") {
    return kCodecNone;
  }
  if (codec == "lzav_default") {
    return kCodecLzavDefault;
  }
  if (codec == "lzav_hi") {
    return kCodecLzavHi;
  }
  throw std::runtime_error("Unsupported RowPack codec: " + std::string(codec));
}

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

template <typename T>
inline void write_exact(std::ofstream& output, T const& value) {
  output.write(reinterpret_cast<char const*>(&value), sizeof(T));
  require(static_cast<bool>(output), "Failed to write RowPack data");
}

inline void write_bytes(std::ofstream& output, std::uint8_t const* data, std::size_t size) {
  if (size == 0) {
    return;
  }
  output.write(reinterpret_cast<char const*>(data), static_cast<std::streamsize>(size));
  require(static_cast<bool>(output), "Failed to write RowPack bytes");
}

inline std::string json_escape(std::string_view value) {
  std::string out;
  out.reserve(value.size() + 2);
  for (char ch : value) {
    switch (ch) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20U) {
          char buffer[7]{};
          std::snprintf(buffer, sizeof(buffer), "\\u%04x", static_cast<unsigned char>(ch));
          out += buffer;
        } else {
          out += ch;
        }
        break;
    }
  }
  return out;
}

inline std::string json_quote(std::string_view value) { return "\"" + json_escape(value) + "\""; }

inline std::string trim_json(std::string_view value) {
  auto begin = value.begin();
  auto end = value.end();
  while (begin != end && std::isspace(static_cast<unsigned char>(*begin)) != 0) {
    ++begin;
  }
  while (begin != end && std::isspace(static_cast<unsigned char>(*(end - 1))) != 0) {
    --end;
  }
  return {begin, end};
}

inline std::string json_object_inner(std::string_view object_json) {
  auto trimmed = trim_json(object_json);
  if (trimmed.empty() || trimmed == "{}") {
    return {};
  }
  require(trimmed.front() == '{' && trimmed.back() == '}', "RowPack metadata_json must be a JSON object");
  return trimmed.substr(1, trimmed.size() - 2);
}

class MetadataBuilder {
 public:
  MetadataBuilder& set_string(std::string key, std::string value) {
    fields_[std::move(key)] = json_quote(value);
    return *this;
  }

  MetadataBuilder& set_json(std::string key, std::string raw_json) {
    fields_[std::move(key)] = std::move(raw_json);
    return *this;
  }

  MetadataBuilder& set_bool(std::string key, bool value) {
    fields_[std::move(key)] = value ? "true" : "false";
    return *this;
  }

  template <typename Number>
  MetadataBuilder& set_number(std::string key, Number value) {
    std::ostringstream out;
    out << value;
    fields_[std::move(key)] = out.str();
    return *this;
  }

  MetadataBuilder& description(std::string value) { return set_string("description", std::move(value)); }
  MetadataBuilder& dataset_name(std::string value) { return set_string("dataset_name", std::move(value)); }
  MetadataBuilder& date_taken(std::string value) { return set_string("date_taken", std::move(value)); }

  MetadataBuilder& add_row_field(std::string name, std::string type, std::string meaning = {}) {
    std::string entry = "{\"name\":" + json_quote(name) + ",\"type\":" + json_quote(type);
    if (!meaning.empty()) {
      entry += ",\"meaning\":" + json_quote(meaning);
    }
    entry += "}";
    row_fields_.push_back(std::move(entry));
    return *this;
  }

  MetadataBuilder& add_sensor(std::string name, std::string type, std::string description = {}, std::string extra_json = "{}") {
    std::string entry = "{\"name\":" + json_quote(name) + ",\"type\":" + json_quote(type);
    if (!description.empty()) {
      entry += ",\"description\":" + json_quote(description);
    }
    auto extra = json_object_inner(extra_json);
    if (!extra.empty()) {
      entry += "," + extra;
    }
    entry += "}";
    sensors_.push_back(std::move(entry));
    return *this;
  }

  MetadataBuilder& set_calibration_json(std::string raw_json) {
    return set_json("calibration", std::move(raw_json));
  }

  MetadataBuilder& set_compression_settings(std::string codec, std::uint64_t rows_per_block) {
    std::ostringstream out;
    out << "{\"block_codec\":" << json_quote(codec) << ",\"rows_per_block\":" << rows_per_block << "}";
    return set_json("compression_settings", out.str());
  }

  MetadataBuilder& set_image_codec_settings(std::string codec, std::string options_json = "{}") {
    std::string value = "{\"codec\":" + json_quote(codec);
    auto options = json_object_inner(options_json);
    if (!options.empty()) {
      value += ",\"options\":{" + options + "}";
    }
    value += "}";
    return set_json("image_codec_settings", std::move(value));
  }

  [[nodiscard]] std::string to_json() const {
    std::string out = "{";
    bool first = true;
    auto append_field = [&](std::string_view key, std::string_view value) {
      if (!first) {
        out += ",";
      }
      first = false;
      out += json_quote(key);
      out += ":";
      out += value;
    };

    for (auto const& [key, value] : fields_) {
      append_field(key, value);
    }
    if (!row_fields_.empty()) {
      append_field("row_schema", json_array(row_fields_));
    }
    if (!sensors_.empty()) {
      append_field("sensors", json_array(sensors_));
    }
    out += "}";
    return out;
  }

 private:
  static std::string json_array(std::vector<std::string> const& values) {
    std::string out = "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (i != 0) {
        out += ",";
      }
      out += values[i];
    }
    out += "]";
    return out;
  }

  std::map<std::string, std::string> fields_;
  std::vector<std::string> row_fields_;
  std::vector<std::string> sensors_;
};

struct WriterOptions {
  std::uint64_t rows_per_block{32};
  std::uint32_t block_codec{kCodecNone};
  std::string payload_format{"bytes"};
  std::string metadata_json{"{}"};
  bool overwrite{false};
};

class Writer {
 public:
  Writer() = default;
  Writer(std::filesystem::path path, WriterOptions options = {}) { open(std::move(path), std::move(options)); }
  ~Writer() {
    try {
      if (!closed_ && output_.is_open()) {
        finish();
      }
    } catch (...) {
    }
  }

  Writer(Writer const&) = delete;
  Writer& operator=(Writer const&) = delete;
  Writer(Writer&&) = default;
  Writer& operator=(Writer&&) = default;

  void open(std::filesystem::path path, WriterOptions options = {}) {
    require(!output_.is_open(), "RowPack Writer is already open");
    require(options.rows_per_block > 0, "rows_per_block must be >= 1");
    require(options.block_codec == kCodecNone || options.block_codec == kCodecLzavDefault || options.block_codec == kCodecLzavHi,
            "Unsupported RowPack block codec");
    if (std::filesystem::exists(path) && !options.overwrite) {
      throw std::runtime_error("Refusing to overwrite existing RowPack file: " + path.string());
    }

    path_ = std::move(path);
    options_ = std::move(options);
    if (path_.has_parent_path()) {
      std::filesystem::create_directories(path_.parent_path());
    }
    output_.open(path_, std::ios::binary | std::ios::trunc);
    require(output_.is_open(), "Failed to open RowPack file for writing");
    auto header = make_empty_header();
    write_exact(output_, header);
    closed_ = false;
  }

  std::uint64_t append_row_bytes(std::uint8_t const* data, std::size_t size, std::string name = {},
                                 std::vector<std::string> aliases = {}) {
    require(!closed_, "Cannot append to a closed RowPack Writer");
    require(output_.is_open(), "RowPack Writer is not open");
    if (pending_rows_.size() >= options_.rows_per_block) {
      flush_block();
    }

    auto const row_id = next_row_id_;
    if (pending_rows_.empty()) {
      pending_start_row_ = row_id;
    }

    auto const block_id = static_cast<std::uint32_t>(blocks_.size());
    auto const row_in_block = static_cast<std::uint32_t>(pending_rows_.size());
    auto const offset = static_cast<std::uint64_t>(pending_payload_.size());
    if (size != 0) {
      pending_payload_.insert(pending_payload_.end(), data, data + size);
    }
    pending_rows_.push_back(RowIndexEntry{offset, static_cast<std::uint64_t>(size), block_id, row_in_block});

    row_names_.push_back(std::move(name));
    if (!aliases.empty()) {
      require(!row_names_.back().empty(), "aliases require a canonical row name");
      for (auto const& alias : aliases) {
        aliases_[alias] = row_names_.back();
      }
    }

    ++next_row_id_;
    return row_id;
  }

  std::uint64_t append_row_bytes(std::vector<std::uint8_t> const& bytes, std::string name = {},
                                 std::vector<std::string> aliases = {}) {
    return append_row_bytes(bytes.data(), bytes.size(), std::move(name), std::move(aliases));
  }

  std::uint64_t append_row_bytes(std::string_view bytes, std::string name = {}, std::vector<std::string> aliases = {}) {
    return append_row_bytes(reinterpret_cast<std::uint8_t const*>(bytes.data()), bytes.size(), std::move(name),
                            std::move(aliases));
  }

#ifdef ROWPACK_USE_CISTA
  std::uint64_t append_cista_row(cast_payload::Row const& row, std::string name = {},
                                 std::vector<std::string> aliases = {}) {
    auto encoded = cista::serialize(row);
    return append_row_bytes(reinterpret_cast<std::uint8_t const*>(encoded.data()), encoded.size(), std::move(name),
                            std::move(aliases));
  }
#endif

  void finish() {
    if (closed_) {
      return;
    }

    flush_block();
    auto const metadata_offset = tellp_u64();
    auto const metadata = build_metadata_json();
    write_bytes(output_, reinterpret_cast<std::uint8_t const*>(metadata.data()), metadata.size());

    auto const block_index_offset = tellp_u64();
    for (auto const& block : blocks_) {
      write_exact(output_, block);
    }
    auto const block_index_size = tellp_u64() - block_index_offset;

    auto const row_index_offset = tellp_u64();
    for (auto const& row : row_index_) {
      write_exact(output_, row);
    }
    auto const row_index_size = tellp_u64() - row_index_offset;

    auto header = make_empty_header();
    header.flags = std::all_of(blocks_.begin(), blocks_.end(), [](auto const& block) { return block.codec == kCodecNone; })
                       ? kFlagUncompressed
                       : 0;
    header.row_count = row_index_.size();
    header.block_count = blocks_.size();
    header.metadata_offset = metadata_offset;
    header.metadata_size = metadata.size();
    header.block_index_offset = block_index_offset;
    header.block_index_size = block_index_size;
    header.row_index_offset = row_index_offset;
    header.row_index_size = row_index_size;

    output_.seekp(0);
    write_exact(output_, header);
    close();
  }

  void close() {
    if (closed_) {
      return;
    }
    closed_ = true;
    output_.close();
  }

 private:
  [[nodiscard]] std::uint64_t tellp_u64() {
    auto pos = output_.tellp();
    require(pos >= 0, "Failed to query RowPack output position");
    return static_cast<std::uint64_t>(pos);
  }

  void flush_block() {
    if (pending_rows_.empty()) {
      return;
    }

    auto payload = pending_payload_;
    auto const codec = options_.block_codec;
    if (codec != kCodecNone) {
      payload = compress_block(pending_payload_, codec);
    }

    auto const block_offset = tellp_u64();
    write_bytes(output_, payload.data(), payload.size());
    auto const block_id = static_cast<std::uint32_t>(blocks_.size());
    blocks_.push_back(BlockIndexEntry{pending_start_row_,
                                      static_cast<std::uint64_t>(pending_rows_.size()),
                                      block_offset,
                                      static_cast<std::uint64_t>(payload.size()),
                                      static_cast<std::uint64_t>(pending_payload_.size()),
                                      codec,
                                      0});

    for (auto entry : pending_rows_) {
      entry.block_id = block_id;
      if (codec == kCodecNone) {
        entry.offset += block_offset;
      }
      row_index_.push_back(entry);
    }

    pending_payload_.clear();
    pending_rows_.clear();
  }

  static std::vector<std::uint8_t> compress_block(std::vector<std::uint8_t> const& input, std::uint32_t codec) {
#ifdef ROWPACK_USE_LZAV
    require(input.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
            "LZAV input block exceeds supported size");
    auto const input_size = static_cast<int>(input.size());
    auto const bound =
        codec == kCodecLzavDefault ? lzav_compress_bound(input_size) : lzav_compress_bound_hi(input_size);
    std::vector<std::uint8_t> out(static_cast<std::size_t>(bound));
    int compressed = 0;
    if (codec == kCodecLzavDefault) {
      compressed = lzav_compress_default(input.data(), out.data(), input_size, bound);
    } else if (codec == kCodecLzavHi) {
      compressed = lzav_compress_hi(input.data(), out.data(), input_size, bound);
    } else {
      throw std::runtime_error("Unsupported RowPack block codec");
    }
    require(compressed != 0 || input.empty(), "LZAV compression failed");
    out.resize(static_cast<std::size_t>(compressed));
    return out;
#else
    (void)input;
    (void)codec;
    throw std::runtime_error("This RowPack build does not include LZAV support");
#endif
  }

  [[nodiscard]] std::string row_names_json() const {
    std::string out = "[";
    for (std::size_t i = 0; i < row_names_.size(); ++i) {
      if (i != 0) {
        out += ",";
      }
      out += row_names_[i].empty() ? "null" : json_quote(row_names_[i]);
    }
    out += "]";
    return out;
  }

  [[nodiscard]] std::string aliases_json() const {
    std::string out = "{";
    bool first = true;
    for (auto const& [alias, canonical] : aliases_) {
      if (!first) {
        out += ",";
      }
      first = false;
      out += json_quote(alias);
      out += ":{\"canonical\":";
      out += json_quote(canonical);
      out += ",\"status\":\"non_canonical\",\"message\":";
      out += json_quote("'" + alias + "' is an alias for canonical row name '" + canonical + "'");
      out += "}";
    }
    out += "}";
    return out;
  }

  [[nodiscard]] std::string observed_compressions_json() const {
    std::vector<std::uint32_t> seen;
    for (auto const& block : blocks_) {
      if (std::find(seen.begin(), seen.end(), block.codec) == seen.end()) {
        seen.push_back(block.codec);
      }
    }
    std::sort(seen.begin(), seen.end());
    std::string out = "[";
    for (std::size_t i = 0; i < seen.size(); ++i) {
      if (i != 0) {
        out += ",";
      }
      out += json_quote(codec_name(seen[i]));
    }
    out += "]";
    return out;
  }

  [[nodiscard]] std::string build_metadata_json() const {
    std::ostringstream out;
    out << "{\"format\":\"RowPack\""
        << ",\"format_version\":\"0.1\""
        << ",\"storage\":\"row-major\""
        << ",\"compression\":" << json_quote(codec_name(options_.block_codec))
        << ",\"block_codec\":" << json_quote(codec_name(options_.block_codec))
        << ",\"observed_compressions\":" << observed_compressions_json()
        << ",\"payload_format\":" << json_quote(options_.payload_format)
        << ",\"rows_per_block\":" << options_.rows_per_block
        << ",\"row_count\":" << row_index_.size()
        << ",\"block_count\":" << blocks_.size()
        << ",\"row_names\":" << row_names_json()
        << ",\"aliases\":" << aliases_json();
    auto extra = json_object_inner(options_.metadata_json);
    if (!extra.empty()) {
      out << "," << extra;
    }
    out << "}";
    return out.str();
  }

  std::filesystem::path path_;
  WriterOptions options_;
  std::ofstream output_;
  bool closed_{true};
  std::uint64_t next_row_id_{0};
  std::uint64_t pending_start_row_{0};
  std::vector<std::uint8_t> pending_payload_;
  std::vector<RowIndexEntry> pending_rows_;
  std::vector<BlockIndexEntry> blocks_;
  std::vector<RowIndexEntry> row_index_;
  std::vector<std::string> row_names_;
  std::map<std::string, std::string> aliases_;
};

#ifdef ROWPACK_USE_CISTA
inline cast_payload::Turn make_turn(std::string role, std::string modality, std::string data) {
  cast_payload::Turn turn;
  turn.role = std::move(role);
  turn.modality = std::move(modality);
  turn.data = std::move(data);
  return turn;
}

inline cast_payload::Image make_image(std::vector<std::uint8_t> bytes, std::uint32_t height = 0,
                                      std::uint32_t width = 0, std::uint32_t channels = 0,
                                      std::string storage = "encoded") {
  cast_payload::Image image;
  image.bytes.resize(bytes.size());
  std::copy(bytes.begin(), bytes.end(), image.bytes.begin());
  image.height = height;
  image.width = width;
  image.channels = channels;
  image.storage = std::move(storage);
  return image;
}
#endif

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
