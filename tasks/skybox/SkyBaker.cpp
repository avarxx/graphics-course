#include "SkyBaker.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>

#include <etna/Etna.hpp>
#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/Profiling.hpp>
#include <imgui.h>
#include <spdlog/spdlog.h>

#include "GpuTimer.hpp"


void SkyBaker::init(std::uint32_t face_size)
{
  auto& ctx = etna::get_context();

  faceSize = std::max(face_size, 8u);
  mips = 1;
  while ((faceSize >> mips) > 0)
    ++mips;

  cube = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{faceSize, faceSize, 1},
    .name = "sky_cubemap",
    .format = FORMAT,
    .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
      vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
    .layers = 6,
    .mipLevels = mips,
    // Allows the cube view for sampling; the bake writes through a 2D array view
    .flags = vk::ImageCreateFlagBits::eCubeCompatible,
  });

  cubeSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .filter = vk::Filter::eLinear,
    .addressMode = vk::SamplerAddressMode::eClampToEdge,
    .mipmapMode = vk::SamplerMipmapMode::eLinear,
    .name = "sky_sampler",
  });

  paramsUbo.emplace(ctx.getMainWorkCount(), [&ctx](std::size_t i) {
    return ctx.createBuffer(etna::Buffer::CreateInfo{
      .size = sizeof(SkyParams),
      .bufferUsage = vk::BufferUsageFlagBits::eUniformBuffer,
      .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
      .allocationCreate =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
      .name = fmt::format("sky_params{}", i),
    });
  });

  spdlog::info("Sky cubemap: 6 x {}x{} R16G16B16A16_SFLOAT, {} mips", faceSize, faceSize, mips);
}

void SkyBaker::loadShaders(const char* shaders_root)
{
  etna::create_program("sky_bake", {std::filesystem::path(shaders_root) / "sky_bake.comp.spv"});
}

void SkyBaker::setupPipelines()
{
  auto& pipelineManager = etna::get_context().getPipelineManager();
  bakePipeline = {};
  bakePipeline = pipelineManager.createComputePipeline("sky_bake", {});
}

void SkyBaker::setSun(glm::vec3 direction, glm::vec3 color, float intensity)
{
  // Exact comparison on purpose: only real changes of the inputs trigger a re-bake
  if (direction != sunDir || color != sunColor || intensity != sunIntensity)
    dirty = true;
  sunDir = direction;
  sunColor = color;
  sunIntensity = intensity;
}

void SkyBaker::updateParams()
{
  params.sunDir = glm::vec4(sunDir, sunIntensity);
  params.sunColor = glm::vec4(sunColor, static_cast<float>(faceSize));
  params.horizonColor = glm::vec4(horizonColor, skyBrightness);
  params.zenithColor = glm::vec4(zenithColor, sunScattering);
  params.sunGlow = glm::vec4(glowNarrow, glowWide, glowNarrowSharpness, glowWideFalloff);
  params.clouds = glm::vec4(coverage, density, cloudScale, cloudsEnabled ? 1.0f : 0.0f);
  params.layer = glm::vec4(
    layerBottom,
    std::max(layerTop, layerBottom + 10.0f),
    static_cast<float>(primarySteps),
    static_cast<float>(lightSteps));
  params.lighting = glm::vec4(cloudAmbient, forwardScattering, erosion, hazeDistance);
  params.offset = glm::vec4(windOffset, 0.0f);
}

void SkyBaker::bakeIfNeeded(vk::CommandBuffer cmd_buf, GpuTimer& timer)
{
  // Nothing changed and no progressive bake is in flight: reuse the baked cubemap
  if (!dirty && nextFace == 0)
    return;

  if (dirty)
  {
    // (Re)start from the first face with the current parameters
    updateParams();
    nextFace = 0;
    dirty = false;
  }

  // The very first bake is always complete: there is no valid content to show otherwise
  const std::uint32_t perFrame = bakes == 0 ? 6u : static_cast<std::uint32_t>(facesPerFrame);
  const std::uint32_t count = std::min(perFrame, 6u - nextFace);

  ETNA_PROFILE_GPU(cmd_buf, skyBake);
  timer.begin(cmd_buf, "Sky bake");

  bakeFaces(cmd_buf, nextFace, count);
  nextFace += count;
  if (nextFace == 6)
  {
    generateMips(cmd_buf);
    nextFace = 0;
    ++bakes;
    spdlog::info("Sky cubemap baked (bake #{})", bakes);
  }

  timer.end(cmd_buf);
}

void SkyBaker::bakeFaces(
  vk::CommandBuffer cmd_buf, std::uint32_t first_face, std::uint32_t face_count)
{
  // One uniform buffer per frame in flight: the previous frame may still be baking on the GPU
  params.offset.w = static_cast<float>(first_face);
  std::memcpy(paramsUbo->get().data(), &params, sizeof(params));

  auto programInfo = etna::get_shader_program("sky_bake");
  auto set = etna::create_descriptor_set(
    programInfo.getDescriptorLayoutId(0),
    cmd_buf,
    {
      // Storage writes go through a 2D array view of mip 0 (6 layers = 6 faces)
      etna::Binding{
        0,
        cube.genBinding(
          {},
          vk::ImageLayout::eGeneral,
          {.baseMip = 0, .levelCount = 1, .type = vk::ImageViewType::e2DArray})},
      etna::Binding{1, paramsUbo->get().genBinding()},
    });
  etna::flush_barriers(cmd_buf);

  cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, bakePipeline.getVkPipeline());
  cmd_buf.bindDescriptorSets(
    vk::PipelineBindPoint::eCompute, bakePipeline.getVkPipelineLayout(), 0, {set.getVkSet()}, {});
  cmd_buf.dispatch((faceSize + 7) / 8, (faceSize + 7) / 8, face_count);
}

void SkyBaker::generateMips(vk::CommandBuffer cmd_buf)
{
  // Etna tracks the state of the whole image. We move the whole image to TRANSFER_DST, build the
  // chain with per-mip barriers by hand, and finally return every mip to TRANSFER_DST, so that
  // etna's idea of the state is correct again (it will transition to SHADER_READ when bound).
  etna::set_state(
    cmd_buf,
    cube.get(),
    vk::PipelineStageFlagBits2::eBlit,
    vk::AccessFlagBits2::eTransferWrite,
    vk::ImageLayout::eTransferDstOptimal,
    vk::ImageAspectFlagBits::eColor);
  etna::flush_barriers(cmd_buf);

  const auto mipBarrier = [&](
                            std::uint32_t mip,
                            std::uint32_t count,
                            vk::AccessFlags2 src_access,
                            vk::AccessFlags2 dst_access,
                            vk::ImageLayout from,
                            vk::ImageLayout to) {
    const vk::ImageMemoryBarrier2 barrier{
      .srcStageMask = vk::PipelineStageFlagBits2::eBlit,
      .srcAccessMask = src_access,
      .dstStageMask = vk::PipelineStageFlagBits2::eBlit,
      .dstAccessMask = dst_access,
      .oldLayout = from,
      .newLayout = to,
      .image = cube.get(),
      .subresourceRange =
        {
          .aspectMask = vk::ImageAspectFlagBits::eColor,
          .baseMipLevel = mip,
          .levelCount = count,
          .baseArrayLayer = 0,
          .layerCount = 6,
        },
    };
    cmd_buf.pipelineBarrier2(vk::DependencyInfo{
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &barrier,
    });
  };

  for (std::uint32_t mip = 1; mip < mips; ++mip)
  {
    mipBarrier(
      mip - 1,
      1,
      vk::AccessFlagBits2::eTransferWrite,
      vk::AccessFlagBits2::eTransferRead,
      vk::ImageLayout::eTransferDstOptimal,
      vk::ImageLayout::eTransferSrcOptimal);

    const auto srcSize = static_cast<std::int32_t>(std::max(faceSize >> (mip - 1), 1u));
    const auto dstSize = static_cast<std::int32_t>(std::max(faceSize >> mip, 1u));
    const vk::ImageBlit blit{
      .srcSubresource = {vk::ImageAspectFlagBits::eColor, mip - 1, 0, 6},
      .srcOffsets = std::array{vk::Offset3D{0, 0, 0}, vk::Offset3D{srcSize, srcSize, 1}},
      .dstSubresource = {vk::ImageAspectFlagBits::eColor, mip, 0, 6},
      .dstOffsets = std::array{vk::Offset3D{0, 0, 0}, vk::Offset3D{dstSize, dstSize, 1}},
    };
    cmd_buf.blitImage(
      cube.get(),
      vk::ImageLayout::eTransferSrcOptimal,
      cube.get(),
      vk::ImageLayout::eTransferDstOptimal,
      {blit},
      vk::Filter::eLinear);
  }

  // Mips [0, mips - 2] are TRANSFER_SRC now, the last one is TRANSFER_DST: unify
  if (mips > 1)
    mipBarrier(
      0,
      mips - 1,
      vk::AccessFlagBits2::eTransferRead,
      vk::AccessFlagBits2::eTransferWrite,
      vk::ImageLayout::eTransferSrcOptimal,
      vk::ImageLayout::eTransferDstOptimal);
}

bool SkyBaker::drawGui()
{
  bool changed = false;

  ImGui::SeparatorText("Sky gradient");
  changed |= ImGui::ColorEdit3("Horizon (light blue)", &horizonColor.x, ImGuiColorEditFlags_Float);
  changed |= ImGui::ColorEdit3("Zenith (dark blue)", &zenithColor.x, ImGuiColorEditFlags_Float);
  changed |= ImGui::SliderFloat("Sky brightness", &skyBrightness, 0.0f, 4.0f);
  changed |= ImGui::SliderFloat(
    "Sun glow (narrow)", &glowNarrow, 0.0f, 50.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
  changed |= ImGui::SliderFloat("Sun halo (wide)", &glowWide, 0.0f, 2.0f);
  changed |= ImGui::SliderFloat(
    "Glow sharpness", &glowNarrowSharpness, 500.0f, 50000.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
  changed |= ImGui::SliderFloat(
    "Halo falloff", &glowWideFalloff, 2.0f, 200.0f, "%.1f", ImGuiSliderFlags_Logarithmic);

  ImGui::SeparatorText("Clouds");
  changed |= ImGui::Checkbox("Clouds", &cloudsEnabled);
  changed |= ImGui::SliderFloat("Cloudiness (coverage)", &coverage, 0.0f, 1.0f);
  changed |= ImGui::SliderFloat(
    "Density (1/m)", &density, 0.001f, 0.2f, "%.4f", ImGuiSliderFlags_Logarithmic);
  changed |= ImGui::SliderFloat(
    "Scale (m)", &cloudScale, 300.0f, 20000.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
  changed |= ImGui::DragFloatRange2(
    "Layer bottom / top (m)", &layerBottom, &layerTop, 10.0f, 100.0f, 12000.0f, "%.0f");
  changed |= ImGui::SliderInt("Primary ray steps", &primarySteps, 4, 256);
  changed |= ImGui::SliderInt("Shadow ray steps", &lightSteps, 0, 16);
  changed |= ImGui::SliderFloat("Detail erosion", &erosion, 0.0f, 1.0f);
  changed |= ImGui::SliderFloat("Forward scattering (g)", &forwardScattering, 0.0f, 0.95f);
  changed |= ImGui::SliderFloat("Sunlight on clouds", &sunScattering, 0.0f, 8.0f);
  changed |= ImGui::SliderFloat("Ambient on clouds", &cloudAmbient, 0.0f, 3.0f);
  changed |= ImGui::SliderFloat(
    "Haze distance (m)", &hazeDistance, 5000.0f, 300000.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
  changed |= ImGui::DragFloat3("Noise offset (wind)", &windOffset.x, 50.0f);

  ImGui::SeparatorText("Baking");
  ImGui::SliderInt("Faces per frame", &facesPerFrame, 1, 6);
  ImGui::SetItemTooltip("Spread one bake over several frames on slow GPUs");
  if (ImGui::Button("Re-bake now"))
    changed = true;

  if (changed)
    dirty = true;
  return changed;
}
