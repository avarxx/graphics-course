#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>


/**
 * CPU generator of an island heightmap using multi-octave gradient noise (fBm).
 * Heights are normalized: 0 is the ocean floor, 1 is the highest possible peak.
 * Everything is deterministic and does not depend on the standard library
 * implementation, so the same seed gives the same island on every platform.
 */
struct HeightmapParams
{
  int seed = 1337;
  int octaves = 7;
  float frequency = 3.0f;     // base noise frequency over the whole map
  float lacunarity = 2.0f;    // frequency multiplier per octave
  float gain = 0.5f;          // amplitude multiplier per octave (persistence)
  float ridge = 0.3f;         // 0 = plain fBm, 1 = fully ridged mountains
  float warp = 0.15f;         // domain warping strength
  float exponent = 1.8f;      // height redistribution, flattens lowlands
  float islandRadius = 0.95f; // radial mask radius in [0, 1.5]
  float islandFalloff = 0.6f; // width of the radial falloff (coastline softness)
};

class HeightmapGenerator
{
public:
  static constexpr std::uint32_t RESOLUTION = 1024;

  // Returns RESOLUTION*RESOLUTION normalized heights, row-major (v * RESOLUTION + u)
  static std::vector<float> generate(const HeightmapParams& params);

  // Bilinear sample of a generated heightmap in uv space ([0,1]^2).
  // Matches the sampling done on the GPU in shaders.
  static float sample(const std::vector<float>& heights, glm::vec2 uv);
};


// Small deterministic hashing utilities, shared with procedural placement code
namespace prng
{

inline std::uint32_t pcg(std::uint32_t v)
{
  std::uint32_t state = v * 747796405u + 2891336453u;
  std::uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

inline std::uint32_t hash(std::uint32_t a, std::uint32_t b, std::uint32_t c = 0)
{
  return pcg(a ^ pcg(b ^ pcg(c)));
}

inline float to01(std::uint32_t h)
{
  return static_cast<float>(h >> 8) * (1.0f / 16777216.0f);
}

} // namespace prng
