#include "HeightmapGenerator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <thread>


namespace
{

constexpr float PI = 3.14159265358979f;

struct GradientNoise
{
  std::array<glm::vec2, 64> gradients;
  std::uint32_t seed;

  explicit GradientNoise(std::uint32_t s)
    : seed{s}
  {
    for (std::size_t i = 0; i < gradients.size(); ++i)
    {
      const float angle =
        2.0f * PI * (static_cast<float>(i) + 0.5f) / static_cast<float>(gradients.size());
      gradients[i] = {std::cos(angle), std::sin(angle)};
    }
  }

  glm::vec2 grad(std::int32_t x, std::int32_t y) const
  {
    const std::uint32_t h =
      prng::hash(static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y), seed);
    return gradients[h & 63u];
  }

  static float fade(float t) { return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); }

  // Classic Perlin-style gradient noise, roughly in [-1, 1]
  float operator()(glm::vec2 p) const
  {
    const float fx = std::floor(p.x);
    const float fy = std::floor(p.y);
    const auto ix = static_cast<std::int32_t>(fx);
    const auto iy = static_cast<std::int32_t>(fy);
    const glm::vec2 f = p - glm::vec2{fx, fy};

    const float n00 = glm::dot(grad(ix, iy), f);
    const float n10 = glm::dot(grad(ix + 1, iy), f - glm::vec2{1, 0});
    const float n01 = glm::dot(grad(ix, iy + 1), f - glm::vec2{0, 1});
    const float n11 = glm::dot(grad(ix + 1, iy + 1), f - glm::vec2{1, 1});

    const float u = fade(f.x);
    const float v = fade(f.y);
    return 1.41421356f * glm::mix(glm::mix(n00, n10, u), glm::mix(n01, n11, u), v);
  }
};

struct Fbm
{
  const GradientNoise& noise;
  const HeightmapParams& params;

  float operator()(glm::vec2 p, int octaves) const
  {
    float sum = 0.0f;
    float amplitude = 1.0f;
    float norm = 0.0f;
    // Rotate every octave a bit to hide grid-aligned artifacts
    const glm::mat2 rot{0.8f, 0.6f, -0.6f, 0.8f};
    for (int i = 0; i < octaves; ++i)
    {
      float n = noise(p);
      // Ridged noise: sharp crests instead of round hills
      const float ridged = 1.0f - 2.0f * std::abs(n);
      n = glm::mix(n, ridged, params.ridge);
      sum += amplitude * n;
      norm += amplitude;
      amplitude *= params.gain;
      p = rot * p * params.lacunarity + glm::vec2{17.13f, -9.71f};
    }
    return norm > 0.0f ? sum / norm : 0.0f;
  }
};

} // namespace

std::vector<float> HeightmapGenerator::generate(const HeightmapParams& params)
{
  constexpr std::uint32_t N = RESOLUTION;
  std::vector<float> raw(static_cast<std::size_t>(N) * N);

  const GradientNoise noise{static_cast<std::uint32_t>(params.seed)};
  const Fbm fbm{noise, params};

  // Seed-dependent offset so that different seeds look completely different
  const glm::vec2 offset{
    prng::to01(prng::hash(static_cast<std::uint32_t>(params.seed), 1u)) * 1000.0f,
    prng::to01(prng::hash(static_cast<std::uint32_t>(params.seed), 2u)) * 1000.0f};

  const int octaves = std::clamp(params.octaves, 1, 16);

  auto worker = [&](std::uint32_t row_begin, std::uint32_t row_end) {
    for (std::uint32_t y = row_begin; y < row_end; ++y)
      for (std::uint32_t x = 0; x < N; ++x)
      {
        const glm::vec2 uv =
          glm::vec2{static_cast<float>(x), static_cast<float>(y)} / static_cast<float>(N - 1);
        glm::vec2 p = uv * params.frequency + offset;

        if (params.warp > 0.0f)
        {
          const glm::vec2 q{
            fbm(p + glm::vec2{5.2f, 1.3f}, 3),
            fbm(p + glm::vec2{1.7f, 9.2f}, 3),
          };
          p += params.warp * 2.0f * q;
        }

        raw[static_cast<std::size_t>(y) * N + x] = fbm(p, octaves);
      }
  };

  const std::uint32_t threadCount =
    std::clamp<std::uint32_t>(std::thread::hardware_concurrency(), 1u, 32u);
  {
    std::vector<std::thread> threads;
    const std::uint32_t rowsPerThread = (N + threadCount - 1) / threadCount;
    for (std::uint32_t t = 0; t < threadCount; ++t)
    {
      const std::uint32_t begin = t * rowsPerThread;
      const std::uint32_t end = std::min(N, begin + rowsPerThread);
      if (begin < end)
        threads.emplace_back(worker, begin, end);
    }
    for (auto& thread : threads)
      thread.join();
  }

  // Normalize noise into [0, 1]
  const auto [minIt, maxIt] = std::minmax_element(raw.begin(), raw.end());
  const float minV = *minIt;
  const float range = std::max(*maxIt - minV, 1e-6f);

  for (std::uint32_t y = 0; y < N; ++y)
    for (std::uint32_t x = 0; x < N; ++x)
    {
      float& h = raw[static_cast<std::size_t>(y) * N + x];
      h = (h - minV) / range;
      h = std::pow(h, params.exponent);

      // Radial island mask
      const glm::vec2 uv =
        glm::vec2{static_cast<float>(x), static_cast<float>(y)} / static_cast<float>(N - 1);
      // The distance is perturbed by low frequency noise to get an irregular coastline
      const glm::vec2 cp = uv * 2.5f + offset * 0.37f;
      const float coastNoise = noise(cp) * 0.6f + noise(cp * 2.3f) * 0.3f;
      const float d = glm::length(uv - 0.5f) * 2.0f + coastNoise * 0.22f;
      const float inner = params.islandRadius - params.islandFalloff;
      const float mask = 1.0f - glm::smoothstep(inner, params.islandRadius, d);

      // Keep some land in the middle even if the noise is low there,
      // and sink everything outside of the mask under the sea.
      h = h * mask + 0.05f * mask;
      h = std::clamp(h, 0.0f, 1.0f);
    }

  return raw;
}

float HeightmapGenerator::sample(const std::vector<float>& heights, glm::vec2 uv)
{
  constexpr std::uint32_t N = RESOLUTION;
  const glm::vec2 p = glm::clamp(uv, 0.0f, 1.0f) * static_cast<float>(N - 1);
  const glm::ivec2 i0 = glm::min(glm::ivec2(glm::floor(p)), glm::ivec2(N - 2));
  const glm::vec2 f = p - glm::vec2(i0);

  auto at = [&](int x, int y) {
    return heights[static_cast<std::size_t>(y) * N + static_cast<std::size_t>(x)];
  };

  const float a = at(i0.x, i0.y);
  const float b = at(i0.x + 1, i0.y);
  const float c = at(i0.x, i0.y + 1);
  const float d = at(i0.x + 1, i0.y + 1);
  return glm::mix(glm::mix(a, b, f.x), glm::mix(c, d, f.x), f.y);
}
