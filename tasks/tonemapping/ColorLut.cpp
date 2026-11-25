#include "ColorLut.hpp"

#include <algorithm>
#include <limits>

#include <spdlog/spdlog.h>


namespace lut
{

namespace
{

constexpr std::uint32_t CELLS = SIZE * SIZE * SIZE;

std::uint32_t cell_index(std::uint32_t r, std::uint32_t g, std::uint32_t b)
{
  return (b * SIZE + g) * SIZE + r;
}

// 8-bit channel value -> cell coordinate: [0, 255] split into SIZE equal ranges of 8 values
std::uint32_t to_cell(std::uint8_t v)
{
  return static_cast<std::uint32_t>(v) * SIZE / 256u;
}

} // namespace

std::optional<std::vector<glm::u8vec4>> build(
  const ImageRGBA8& source, const ImageRGBA8& graded, BuildStats* stats)
{
  if (source.width != graded.width || source.height != graded.height || source.empty())
  {
    spdlog::warn(
      "LUT: source ({}x{}) and graded ({}x{}) images must have the same size",
      source.width,
      source.height,
      graded.width,
      graded.height);
    return std::nullopt;
  }

  // 1-3. Every pixel of the source picks a cell, the cell averages the graded colors
  std::vector<glm::dvec3> sums(CELLS, glm::dvec3(0.0));
  std::vector<std::uint32_t> counts(CELLS, 0);
  for (std::uint32_t y = 0; y < source.height; ++y)
    for (std::uint32_t x = 0; x < source.width; ++x)
    {
      const glm::u8vec4 s = source.at(x, y);
      const glm::u8vec4 g = graded.at(x, y);
      const std::uint32_t c = cell_index(to_cell(s.r), to_cell(s.g), to_cell(s.b));
      sums[c] += glm::dvec3(g.r, g.g, g.b);
      ++counts[c];
    }

  std::vector<glm::u8vec4> texels(CELLS);
  std::vector<glm::ivec3> filled;
  filled.reserve(CELLS);
  for (std::uint32_t b = 0; b < SIZE; ++b)
    for (std::uint32_t g = 0; g < SIZE; ++g)
      for (std::uint32_t r = 0; r < SIZE; ++r)
      {
        const std::uint32_t c = cell_index(r, g, b);
        if (counts[c] == 0)
          continue;
        const glm::dvec3 avg = sums[c] / static_cast<double>(counts[c]);
        texels[c] = glm::u8vec4(glm::clamp(glm::round(avg), 0.0, 255.0), 255);
        filled.emplace_back(r, g, b);
      }

  if (filled.empty())
    return std::nullopt;

  // 4. Empty cells take the color of the nearest filled cell
  for (std::uint32_t b = 0; b < SIZE; ++b)
    for (std::uint32_t g = 0; g < SIZE; ++g)
      for (std::uint32_t r = 0; r < SIZE; ++r)
      {
        const std::uint32_t c = cell_index(r, g, b);
        if (counts[c] != 0)
          continue;
        const glm::ivec3 p(r, g, b);
        int best = std::numeric_limits<int>::max();
        glm::ivec3 nearest = filled.front();
        for (const glm::ivec3& f : filled)
        {
          const glm::ivec3 d = f - p;
          const int dist = d.x * d.x + d.y * d.y + d.z * d.z;
          if (dist < best)
          {
            best = dist;
            nearest = f;
          }
        }
        texels[c] = texels[cell_index(
          static_cast<std::uint32_t>(nearest.x),
          static_cast<std::uint32_t>(nearest.y),
          static_cast<std::uint32_t>(nearest.z))];
      }

  if (stats != nullptr)
  {
    stats->filledCells = static_cast<std::uint32_t>(filled.size());
    stats->pixels = source.width * source.height;
  }
  return texels;
}

std::vector<glm::u8vec4> identity()
{
  std::vector<glm::u8vec4> texels(CELLS);
  for (std::uint32_t b = 0; b < SIZE; ++b)
    for (std::uint32_t g = 0; g < SIZE; ++g)
      for (std::uint32_t r = 0; r < SIZE; ++r)
      {
        // Center of the cell: the average color of the 8 values that fall into it
        const auto center = [](std::uint32_t v) {
          return static_cast<std::uint8_t>(v * 256u / SIZE + 256u / SIZE / 2u);
        };
        texels[cell_index(r, g, b)] = glm::u8vec4(center(r), center(g), center(b), 255);
      }
  return texels;
}

std::vector<Entry> load_all(const std::filesystem::path& dir)
{
  std::vector<Entry> result;
  std::error_code ec;
  if (!std::filesystem::is_directory(dir, ec))
  {
    spdlog::warn("LUT folder {} does not exist", dir.string());
    return result;
  }

  std::vector<std::filesystem::path> folders;
  for (const auto& entry : std::filesystem::directory_iterator(dir, ec))
    if (entry.is_directory())
      folders.push_back(entry.path());
  std::sort(folders.begin(), folders.end());

  for (const auto& folder : folders)
  {
    auto source = tex::find_image(folder, "source");
    if (!source)
      source = tex::find_image(dir, "source");
    const auto graded = tex::find_image(folder, "graded");
    if (!source || !graded)
      continue;
    const auto src = tex::load_rgba8(*source);
    const auto grd = tex::load_rgba8(*graded);
    if (!src || !grd)
      continue;

    Entry e;
    e.name = folder.filename().string();
    auto texels = build(*src, *grd, &e.stats);
    if (!texels)
      continue;
    e.texels = std::move(*texels);
    spdlog::info(
      "LUT '{}': {} pixels, {} / {} cells filled from the images",
      e.name,
      e.stats.pixels,
      e.stats.filledCells,
      CELLS);
    result.push_back(std::move(e));
  }
  return result;
}

} // namespace lut
