#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "TextureLoader.hpp"


/**
 * Building a 3D color-grading LUT from an image and its color-corrected version.
 *
 * The RGB cube is split into SIZE^3 cells. Every pixel of the source image selects a cell by its
 * color, and the cell accumulates the color of the same pixel of the graded image. Each cell gets
 * the average; empty cells copy the nearest (Euclidean distance in cell space) filled cell.
 */
namespace lut
{

inline constexpr std::uint32_t SIZE = 32;

struct BuildStats
{
  std::uint32_t filledCells = 0;
  std::uint32_t pixels = 0;
};

// Returns SIZE^3 RGBA8 texels, x = red is the fastest axis, then green, then blue
// (the layout of a 3D image with width = red, height = green, depth = blue)
std::optional<std::vector<glm::u8vec4>> build(
  const ImageRGBA8& source, const ImageRGBA8& graded, BuildStats* stats = nullptr);

std::vector<glm::u8vec4> identity();

struct Entry
{
  std::string name;
  std::vector<glm::u8vec4> texels;
  BuildStats stats;
};

// Every subfolder of `dir` with a `graded.png` becomes a LUT. The source image is
// `<subfolder>/source.png` if present, otherwise the shared `<dir>/source.png`.
std::vector<Entry> load_all(const std::filesystem::path& dir);

} // namespace lut
