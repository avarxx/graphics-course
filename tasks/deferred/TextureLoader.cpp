#include "TextureLoader.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <etna/Etna.hpp>
#include <etna/GlobalContext.hpp>
#include <spdlog/spdlog.h>
#include <stb_image.h>


namespace tex
{

std::optional<std::filesystem::path> find_image(
  const std::filesystem::path& dir, std::string_view name)
{
  for (const char* ext : {".jpg", ".png", ".jpeg"})
  {
    auto path = dir / (std::string(name) + ext);
    if (std::filesystem::exists(path))
      return path;
  }
  return std::nullopt;
}

std::optional<ImageRGBA8> load_rgba8(const std::filesystem::path& path)
{
  int w = 0;
  int h = 0;
  int comp = 0;
  stbi_uc* data = stbi_load(path.string().c_str(), &w, &h, &comp, 4);
  if (data == nullptr)
  {
    spdlog::warn("Failed to load image {}: {}", path.string(), stbi_failure_reason());
    return std::nullopt;
  }

  ImageRGBA8 result;
  result.width = static_cast<std::uint32_t>(w);
  result.height = static_cast<std::uint32_t>(h);
  result.pixels.assign(data, data + static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4);
  stbi_image_free(data);
  return result;
}

ImageRGBA8 make_solid(std::uint32_t width, std::uint32_t height, glm::u8vec4 color)
{
  ImageRGBA8 result{width, height, {}};
  result.pixels.resize(static_cast<std::size_t>(width) * height * 4);
  for (std::uint32_t y = 0; y < height; ++y)
    for (std::uint32_t x = 0; x < width; ++x)
      result.at(x, y) = color;
  return result;
}

ImageRGBA8 from_components(
  std::uint32_t width, std::uint32_t height, int components, std::span<const std::uint8_t> data)
{
  ImageRGBA8 result = make_solid(width, height, {0, 0, 0, 255});
  const auto comps = static_cast<std::size_t>(std::clamp(components, 1, 4));
  for (std::uint32_t y = 0; y < height; ++y)
    for (std::uint32_t x = 0; x < width; ++x)
    {
      const std::size_t idx = (static_cast<std::size_t>(y) * width + x) * comps;
      glm::u8vec4 px{0, 0, 0, 255};
      if (idx + comps <= data.size())
      {
        if (comps == 1)
          px = {data[idx], data[idx], data[idx], 255};
        else if (comps == 2)
          px = {data[idx], data[idx], data[idx], data[idx + 1]};
        else
          for (std::size_t c = 0; c < comps; ++c)
            px[static_cast<glm::length_t>(c)] = data[idx + c];
      }
      result.at(x, y) = px;
    }
  return result;
}

ImageRGBA8 resize(const ImageRGBA8& src, std::uint32_t width, std::uint32_t height)
{
  if (src.width == width && src.height == height)
    return src;

  ImageRGBA8 dst = make_solid(width, height, {0, 0, 0, 255});
  for (std::uint32_t y = 0; y < height; ++y)
    for (std::uint32_t x = 0; x < width; ++x)
    {
      const float u =
        (static_cast<float>(x) + 0.5f) / static_cast<float>(width) * static_cast<float>(src.width) -
        0.5f;
      const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(height) *
          static_cast<float>(src.height) -
        0.5f;
      const auto x0 = static_cast<std::uint32_t>(
        std::clamp(std::floor(u), 0.0f, static_cast<float>(src.width - 1)));
      const auto y0 = static_cast<std::uint32_t>(
        std::clamp(std::floor(v), 0.0f, static_cast<float>(src.height - 1)));
      const std::uint32_t x1 = std::min(x0 + 1, src.width - 1);
      const std::uint32_t y1 = std::min(y0 + 1, src.height - 1);
      const float fx = std::clamp(u - static_cast<float>(x0), 0.0f, 1.0f);
      const float fy = std::clamp(v - static_cast<float>(y0), 0.0f, 1.0f);

      const glm::vec4 a = glm::vec4(src.at(x0, y0));
      const glm::vec4 b = glm::vec4(src.at(x1, y0));
      const glm::vec4 c = glm::vec4(src.at(x0, y1));
      const glm::vec4 d = glm::vec4(src.at(x1, y1));
      const glm::vec4 r = glm::mix(glm::mix(a, b, fx), glm::mix(c, d, fx), fy);
      dst.at(x, y) = glm::u8vec4(glm::clamp(r + 0.5f, 0.0f, 255.0f));
    }
  return dst;
}

std::uint32_t mip_count(std::uint32_t width, std::uint32_t height)
{
  std::uint32_t levels = 1;
  std::uint32_t size = std::max(width, height);
  while (size > 1)
  {
    size /= 2;
    ++levels;
  }
  return levels;
}

namespace
{

float srgb_to_linear(float c)
{
  return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

float linear_to_srgb(float c)
{
  return c <= 0.0031308f ? c * 12.92f : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

ImageRGBA8 downsample(const ImageRGBA8& src, bool srgb)
{
  const std::uint32_t w = std::max(1u, src.width / 2);
  const std::uint32_t h = std::max(1u, src.height / 2);
  ImageRGBA8 dst = make_solid(w, h, {0, 0, 0, 255});

  for (std::uint32_t y = 0; y < h; ++y)
    for (std::uint32_t x = 0; x < w; ++x)
    {
      glm::vec4 sum{0.0f};
      for (std::uint32_t dy = 0; dy < 2; ++dy)
        for (std::uint32_t dx = 0; dx < 2; ++dx)
        {
          const std::uint32_t sx = std::min(x * 2 + dx, src.width - 1);
          const std::uint32_t sy = std::min(y * 2 + dy, src.height - 1);
          glm::vec4 px = glm::vec4(src.at(sx, sy)) / 255.0f;
          if (srgb)
            px = {srgb_to_linear(px.r), srgb_to_linear(px.g), srgb_to_linear(px.b), px.a};
          sum += px;
        }
      glm::vec4 avg = sum * 0.25f;
      if (srgb)
        avg = {linear_to_srgb(avg.r), linear_to_srgb(avg.g), linear_to_srgb(avg.b), avg.a};
      dst.at(x, y) = glm::u8vec4(glm::clamp(avg * 255.0f + 0.5f, 0.0f, 255.0f));
    }
  return dst;
}

bool is_srgb(vk::Format format)
{
  return format == vk::Format::eR8G8B8A8Srgb || format == vk::Format::eB8G8R8A8Srgb;
}

} // namespace

etna::Image create_texture(
  etna::OneShotCmdMgr& cmd_mgr,
  std::string_view name,
  vk::Format format,
  std::span<const ImageRGBA8> layers)
{
  ETNA_VERIFY(!layers.empty());
  const std::uint32_t width = layers[0].width;
  const std::uint32_t height = layers[0].height;
  for (const auto& layer : layers)
    ETNA_VERIFYF(
      layer.width == width && layer.height == height,
      "All layers of texture {} must have the same size!",
      name);

  const std::uint32_t levels = mip_count(width, height);
  const bool srgb = is_srgb(format);

  // Build all mip levels for all layers and pack them into one staging buffer
  struct Region
  {
    std::size_t offset;
    std::uint32_t layer;
    std::uint32_t mip;
    std::uint32_t w;
    std::uint32_t h;
  };
  std::vector<Region> regions;
  std::vector<std::uint8_t> staging;

  for (std::uint32_t layer = 0; layer < layers.size(); ++layer)
  {
    ImageRGBA8 current = layers[layer];
    for (std::uint32_t mip = 0; mip < levels; ++mip)
    {
      regions.push_back(Region{staging.size(), layer, mip, current.width, current.height});
      staging.insert(staging.end(), current.pixels.begin(), current.pixels.end());
      if (mip + 1 < levels)
        current = downsample(current, srgb);
    }
  }

  auto& ctx = etna::get_context();

  etna::Image image = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{width, height, 1},
    .name = name,
    .format = format,
    .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
    .layers = layers.size(),
    .mipLevels = levels,
  });

  etna::Buffer stagingBuffer = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = staging.size(),
    .bufferUsage = vk::BufferUsageFlagBits::eTransferSrc,
    .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
    .allocationCreate =
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
    .name = "texture_staging",
  });
  std::memcpy(stagingBuffer.data(), staging.data(), staging.size());

  std::vector<vk::BufferImageCopy2> copies;
  copies.reserve(regions.size());
  for (const auto& r : regions)
    copies.push_back(vk::BufferImageCopy2{
      .bufferOffset = r.offset,
      .bufferRowLength = 0,
      .bufferImageHeight = 0,
      .imageSubresource =
        vk::ImageSubresourceLayers{
          .aspectMask = vk::ImageAspectFlagBits::eColor,
          .mipLevel = r.mip,
          .baseArrayLayer = r.layer,
          .layerCount = 1,
        },
      .imageOffset = vk::Offset3D{0, 0, 0},
      .imageExtent = vk::Extent3D{r.w, r.h, 1},
    });

  auto cmdBuf = cmd_mgr.start();
  ETNA_CHECK_VK_RESULT(cmdBuf.begin(vk::CommandBufferBeginInfo{}));
  {
    etna::set_state(
      cmdBuf,
      image.get(),
      vk::PipelineStageFlagBits2::eTransfer,
      vk::AccessFlagBits2::eTransferWrite,
      vk::ImageLayout::eTransferDstOptimal,
      vk::ImageAspectFlagBits::eColor);
    etna::flush_barriers(cmdBuf);

    cmdBuf.copyBufferToImage2(vk::CopyBufferToImageInfo2{
      .srcBuffer = stagingBuffer.get(),
      .dstImage = image.get(),
      .dstImageLayout = vk::ImageLayout::eTransferDstOptimal,
      .regionCount = static_cast<std::uint32_t>(copies.size()),
      .pRegions = copies.data(),
    });

    etna::set_state(
      cmdBuf,
      image.get(),
      vk::PipelineStageFlagBits2::eFragmentShader,
      vk::AccessFlagBits2::eShaderSampledRead,
      vk::ImageLayout::eShaderReadOnlyOptimal,
      vk::ImageAspectFlagBits::eColor);
    etna::flush_barriers(cmdBuf);
  }
  ETNA_CHECK_VK_RESULT(cmdBuf.end());
  cmd_mgr.submitAndWait(std::move(cmdBuf));

  return image;
}

void upload_raw(etna::OneShotCmdMgr& cmd_mgr, etna::Image& image, std::span<const std::byte> data)
{
  auto& ctx = etna::get_context();
  const auto extent = image.getExtent();

  etna::Buffer stagingBuffer = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = data.size(),
    .bufferUsage = vk::BufferUsageFlagBits::eTransferSrc,
    .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
    .allocationCreate =
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
    .name = "raw_staging",
  });
  std::memcpy(stagingBuffer.data(), data.data(), data.size());

  const vk::BufferImageCopy2 copy{
    .bufferOffset = 0,
    .bufferRowLength = 0,
    .bufferImageHeight = 0,
    .imageSubresource =
      vk::ImageSubresourceLayers{
        .aspectMask = image.getAspectMaskByFormat(),
        .mipLevel = 0,
        .baseArrayLayer = 0,
        .layerCount = 1,
      },
    .imageOffset = vk::Offset3D{0, 0, 0},
    .imageExtent = extent,
  };

  auto cmdBuf = cmd_mgr.start();
  ETNA_CHECK_VK_RESULT(cmdBuf.begin(vk::CommandBufferBeginInfo{}));
  {
    etna::set_state(
      cmdBuf,
      image.get(),
      vk::PipelineStageFlagBits2::eTransfer,
      vk::AccessFlagBits2::eTransferWrite,
      vk::ImageLayout::eTransferDstOptimal,
      image.getAspectMaskByFormat());
    etna::flush_barriers(cmdBuf);

    cmdBuf.copyBufferToImage2(vk::CopyBufferToImageInfo2{
      .srcBuffer = stagingBuffer.get(),
      .dstImage = image.get(),
      .dstImageLayout = vk::ImageLayout::eTransferDstOptimal,
      .regionCount = 1,
      .pRegions = &copy,
    });

    etna::set_state(
      cmdBuf,
      image.get(),
      vk::PipelineStageFlagBits2::eVertexShader | vk::PipelineStageFlagBits2::eFragmentShader |
        vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderSampledRead,
      vk::ImageLayout::eShaderReadOnlyOptimal,
      image.getAspectMaskByFormat());
    etna::flush_barriers(cmdBuf);
  }
  ETNA_CHECK_VK_RESULT(cmdBuf.end());
  cmd_mgr.submitAndWait(std::move(cmdBuf));
}

} // namespace tex
