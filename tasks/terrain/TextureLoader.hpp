#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <etna/Image.hpp>
#include <etna/OneShotCmdMgr.hpp>
#include <glm/glm.hpp>


struct ImageRGBA8
{
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> pixels; // width * height * 4

  bool empty() const { return pixels.empty(); }
  glm::u8vec4& at(std::uint32_t x, std::uint32_t y)
  {
    return reinterpret_cast<glm::u8vec4*>(pixels.data())[static_cast<std::size_t>(y) * width + x];
  }
  const glm::u8vec4& at(std::uint32_t x, std::uint32_t y) const
  {
    return reinterpret_cast<const glm::u8vec4*>(
      pixels.data())[static_cast<std::size_t>(y) * width + x];
  }
};

namespace tex
{

// Returns "<dir>/<name>.jpg" or "<dir>/<name>.png", whichever exists
std::optional<std::filesystem::path> find_image(
  const std::filesystem::path& dir, std::string_view name);

std::optional<ImageRGBA8> load_rgba8(const std::filesystem::path& path);

ImageRGBA8 make_solid(std::uint32_t width, std::uint32_t height, glm::u8vec4 color);

ImageRGBA8 from_components(
  std::uint32_t width, std::uint32_t height, int components, std::span<const std::uint8_t> data);

ImageRGBA8 resize(const ImageRGBA8& src, std::uint32_t width, std::uint32_t height);

std::uint32_t mip_count(std::uint32_t width, std::uint32_t height);

/**
 * Creates a (possibly layered) 2D texture with a full mip chain and uploads it.
 * All layers must have the same size. Mips are generated on the CPU with a box
 * filter (in linear space for sRGB formats).
 */
etna::Image create_texture(
  etna::OneShotCmdMgr& cmd_mgr,
  std::string_view name,
  vk::Format format,
  std::span<const ImageRGBA8> layers);

// Uploads raw bytes into mip 0 / layer 0 of an existing image and transitions it for sampling.
void upload_raw(
  etna::OneShotCmdMgr& cmd_mgr, etna::Image& image, std::span<const std::byte> data);

} // namespace tex
