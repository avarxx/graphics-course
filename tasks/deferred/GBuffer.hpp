#pragma once

#include <array>
#include <vector>

#include <etna/Image.hpp>
#include <glm/glm.hpp>


enum class GBufferLayout
{
  Compact = 0, // 14 bytes per pixel, see shaders/gbuffer.glsl
  Naive = 1,   // 28 bytes per pixel, everything in RGBA16F
};

/**
 * G-buffer render targets: albedo, normal, material parameters and depth.
 * The world position is never stored, it is restored from depth.
 */
class GBuffer
{
public:
  static constexpr std::size_t COLOR_TARGETS = 3;
  static constexpr vk::Format DEPTH_FORMAT = vk::Format::eD32Sfloat;

  static std::array<vk::Format, COLOR_TARGETS> colorFormats(GBufferLayout gb_layout);
  static std::uint32_t bytesPerPixel(GBufferLayout gb_layout);

  void allocate(glm::uvec2 new_resolution, GBufferLayout gb_layout);

  GBufferLayout layout() const { return currentLayout; }
  glm::uvec2 resolution() const { return res; }

  etna::Image albedo;
  etna::Image normal;
  etna::Image material;
  etna::Image depth;

  std::array<const etna::Image*, COLOR_TARGETS> colorTargets() const
  {
    return {&albedo, &normal, &material};
  }

private:
  GBufferLayout currentLayout = GBufferLayout::Compact;
  glm::uvec2 res{};
};
