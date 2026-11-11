#include "GBuffer.hpp"

#include <etna/GlobalContext.hpp>


std::array<vk::Format, GBuffer::COLOR_TARGETS> GBuffer::colorFormats(GBufferLayout gb_layout)
{
  if (gb_layout == GBufferLayout::Compact)
    return {
      // rgb = albedo, a = ambient occlusion
      vk::Format::eR8G8B8A8Srgb,
      // octahedral normal. RG16F is guaranteed to support color attachment + blending
      // (unlike RG16_SNORM), and has ~11 bits of precision in [-1, 1], enough for normals.
      vk::Format::eR16G16Sfloat,
      // r = roughness, g = metallic
      vk::Format::eR8G8Unorm,
    };
  return {
    vk::Format::eR16G16B16A16Sfloat,
    vk::Format::eR16G16B16A16Sfloat,
    vk::Format::eR16G16B16A16Sfloat,
  };
}

std::uint32_t GBuffer::bytesPerPixel(GBufferLayout gb_layout)
{
  // Depth (D32) is included
  return gb_layout == GBufferLayout::Compact ? 4 + 4 + 2 + 4 : 8 + 8 + 8 + 4;
}

void GBuffer::allocate(glm::uvec2 new_resolution, GBufferLayout gb_layout)
{
  auto& ctx = etna::get_context();
  currentLayout = gb_layout;
  res = new_resolution;

  const auto formats = colorFormats(gb_layout);
  const char* names[COLOR_TARGETS] = {"gbuffer_albedo", "gbuffer_normal", "gbuffer_material"};
  etna::Image* targets[COLOR_TARGETS] = {&albedo, &normal, &material};
  for (std::size_t i = 0; i < COLOR_TARGETS; ++i)
    *targets[i] = ctx.createImage(etna::Image::CreateInfo{
      .extent = vk::Extent3D{new_resolution.x, new_resolution.y, 1},
      .name = names[i],
      .format = formats[i],
      .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
    });

  depth = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{new_resolution.x, new_resolution.y, 1},
    .name = "gbuffer_depth",
    .format = DEPTH_FORMAT,
    .imageUsage =
      vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
  });
}
