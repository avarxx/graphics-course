#include "Tonemapper.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>

#include <etna/Etna.hpp>
#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/Profiling.hpp>
#include <etna/RenderTargetStates.hpp>
#include <imgui.h>
#include <spdlog/spdlog.h>

#include "ColorLut.hpp"
#include "GpuTimer.hpp"


namespace
{

// Etna only tracks images, the state buffer is synchronized by hand
void memory_barrier(
  vk::CommandBuffer cmd_buf,
  vk::PipelineStageFlags2 src_stage,
  vk::AccessFlags2 src_access,
  vk::PipelineStageFlags2 dst_stage,
  vk::AccessFlags2 dst_access)
{
  const vk::MemoryBarrier2 barrier{
    .srcStageMask = src_stage,
    .srcAccessMask = src_access,
    .dstStageMask = dst_stage,
    .dstAccessMask = dst_access,
  };
  cmd_buf.pipelineBarrier2(vk::DependencyInfo{
    .memoryBarrierCount = 1,
    .pMemoryBarriers = &barrier,
  });
}

constexpr auto COMPUTE = vk::PipelineStageFlagBits2::eComputeShader;
constexpr auto STORAGE_RW =
  vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite;

} // namespace

void Tonemapper::init(etna::OneShotCmdMgr& cmd_mgr)
{
  auto& ctx = etna::get_context();
  cmdMgr = &cmd_mgr;

  stateBuffer = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = sizeof(TonemapState),
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
      vk::BufferUsageFlagBits::eTransferSrc,
    .name = "tonemap_state",
  });

  readbackBuffers.emplace(ctx.getMainWorkCount(), [&ctx](std::size_t i) {
    return ctx.createBuffer(etna::Buffer::CreateInfo{
      .size = sizeof(TonemapState),
      .bufferUsage = vk::BufferUsageFlagBits::eTransferDst,
      .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
      .allocationCreate =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
      .name = fmt::format("tonemap_readback{}", i),
    });
  });
  // Nothing has been copied into them yet
  readbackBuffers->iterate([](etna::Buffer& b) { std::memset(b.data(), 0, sizeof(TonemapState)); });

  // Zero the whole state once: `initialized` = 0 makes the first frame adapt immediately
  {
    auto cmd = cmd_mgr.start();
    ETNA_CHECK_VK_RESULT(cmd.begin(vk::CommandBufferBeginInfo{}));
    cmd.fillBuffer(stateBuffer.get(), 0, VK_WHOLE_SIZE, 0);
    ETNA_CHECK_VK_RESULT(cmd.end());
    cmd_mgr.submitAndWait(cmd);
  }

  pointSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .filter = vk::Filter::eNearest,
    .addressMode = vk::SamplerAddressMode::eClampToEdge,
    .mipmapMode = vk::SamplerMipmapMode::eNearest,
    .name = "tonemap_point_sampler",
  });
  // Trilinear filtering between LUT cells, clamped at the edges of the RGB cube
  lutSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .filter = vk::Filter::eLinear,
    .addressMode = vk::SamplerAddressMode::eClampToEdge,
    .mipmapMode = vk::SamplerMipmapMode::eNearest,
    .name = "lut_sampler",
  });

  lutImage = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{lut::SIZE, lut::SIZE, lut::SIZE},
    .name = "grading_lut",
    .format = vk::Format::eR8G8B8A8Unorm,
    .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
    .type = vk::ImageType::e3D,
  });

  luts.push_back(LutEntry{.name = "identity", .texels = lut::identity(), .filledCells = 0});
  for (auto& e : lut::load_all(TONEMAPPING_LUTS_ROOT))
    luts.push_back(
      LutEntry{.name = e.name, .texels = std::move(e.texels), .filledCells = e.stats.filledCells});

  // A graded LUT by default if there is one
  lutIndex = luts.size() > 1 ? 1 : 0;
  selectLut("film_warm");
  uploadLut(cmd_mgr, static_cast<std::size_t>(lutIndex));
}

bool Tonemapper::selectLut(const std::string& name)
{
  for (std::size_t i = 0; i < luts.size(); ++i)
    if (luts[i].name == name)
    {
      lutIndex = static_cast<int>(i);
      return true;
    }
  spdlog::warn("LUT '{}' not found", name);
  return false;
}

void Tonemapper::uploadLut(etna::OneShotCmdMgr& cmd_mgr, std::size_t index)
{
  if (index >= luts.size() || index == uploadedLut)
    return;
  // The LUT may be in use by the frames in flight
  ETNA_CHECK_VK_RESULT(etna::get_context().getDevice().waitIdle());
  const auto& texels = luts[index].texels;
  tex::upload_raw(
    cmd_mgr, lutImage, std::as_bytes(std::span<const glm::u8vec4>(texels.data(), texels.size())));
  uploadedLut = index;
}

void Tonemapper::loadShaders(const char* shaders_root)
{
  const auto shader = [shaders_root](const char* name) {
    return std::filesystem::path(shaders_root) / name;
  };
  etna::create_program("lum_minmax_global", {shader("minmax_global.comp.spv")});
  etna::create_program("lum_minmax_shared", {shader("minmax_shared.comp.spv")});
  etna::create_program("lum_histogram_global", {shader("histogram_global.comp.spv")});
  etna::create_program("lum_histogram_shared", {shader("histogram_shared.comp.spv")});
  etna::create_program("exposure", {shader("exposure.comp.spv")});
  etna::create_program(
    "postprocess", {shader("fullscreen.vert.spv"), shader("postprocess.frag.spv")});
}

void Tonemapper::setupPipelines(vk::Format swapchain_format)
{
  auto& pm = etna::get_context().getPipelineManager();
  minmaxPipelines[0] = {};
  minmaxPipelines[0] = pm.createComputePipeline("lum_minmax_global", {});
  minmaxPipelines[1] = {};
  minmaxPipelines[1] = pm.createComputePipeline("lum_minmax_shared", {});
  histogramPipelines[0] = {};
  histogramPipelines[0] = pm.createComputePipeline("lum_histogram_global", {});
  histogramPipelines[1] = {};
  histogramPipelines[1] = pm.createComputePipeline("lum_histogram_shared", {});
  exposurePipeline = {};
  exposurePipeline = pm.createComputePipeline("exposure", {});

  postPipeline = {};
  postPipeline = pm.createGraphicsPipeline(
    "postprocess",
    etna::GraphicsPipeline::CreateInfo{
      .rasterizationConfig =
        vk::PipelineRasterizationStateCreateInfo{
          .polygonMode = vk::PolygonMode::eFill,
          .cullMode = vk::CullModeFlagBits::eNone,
          .frontFace = vk::FrontFace::eCounterClockwise,
          .lineWidth = 1.f,
        },
      .fragmentShaderOutput = {.colorAttachmentFormats = {swapchain_format}},
    });
}

TonemapPush Tonemapper::makePush(glm::uvec2 resolution, float dt) const
{
  return TonemapPush{
    .frame = glm::vec4(
      static_cast<float>(resolution.x),
      static_cast<float>(resolution.y),
      // A long hitch (e.g. a re-bake of the sky) must not make the adaptation jump
      std::min(dt, 0.1f),
      static_cast<float>(exposureMethod)),
    .range = glm::vec4(lowPercentile, highPercentile, minLogLum, maxLogLum),
    .adapt =
      glm::vec4(adaptSpeedUp, adaptSpeedDown, keyValue * std::exp2(exposureCompensation), 0.0f),
  };
}

void Tonemapper::update()
{
  // A different LUT was selected: upload it before the frame's command buffer is recorded
  // (the upload uses its own one-shot command buffer and waits for the GPU)
  if (cmdMgr != nullptr)
    uploadLut(*cmdMgr, static_cast<std::size_t>(lutIndex));
  readStatistics();
}

void Tonemapper::readStatistics()
{
  // This slot was last written by the frame that previously used it, which etna has already
  // waited for. The barrier to the host is recorded after the copy.
  std::memcpy(&stats, readbackBuffers->get().data(), sizeof(TonemapState));
  for (std::size_t i = 0; i < HIST_BINS; ++i)
    histogramPlot[i] = static_cast<float>(stats.histogram[i]);
}

void Tonemapper::computeExposure(
  vk::CommandBuffer cmd_buf,
  const etna::Image& hdr,
  glm::uvec2 resolution,
  float dt,
  GpuTimer& timer)
{
  ETNA_PROFILE_GPU(cmd_buf, autoExposure);
  const std::size_t variant = useSharedMemory ? 1 : 0;
  const TonemapPush push = makePush(resolution, dt);
  const glm::uvec2 groups =
    (resolution + glm::uvec2(LUM_GROUP_SIZE - 1)) / glm::uvec2(LUM_GROUP_SIZE);

  // The previous frame's post-processing (fragment) and readback copy (transfer) are done with
  // the buffer before we clear it
  memory_barrier(
    cmd_buf,
    vk::PipelineStageFlagBits2::eFragmentShader | COMPUTE | vk::PipelineStageFlagBits2::eTransfer,
    vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eTransferRead,
    vk::PipelineStageFlagBits2::eAllTransfer,
    vk::AccessFlagBits2::eTransferWrite);

  // Reset min (to the largest ordered value), max and the histogram; keep the adaptation state.
  // NOTE: vkCmdFillBuffer counts as a COPY (transfer), not CLEAR, so the barriers use ALL_TRANSFER
  cmd_buf.fillBuffer(stateBuffer.get(), 0, 4, 0xFFFFFFFFu);
  cmd_buf.fillBuffer(
    stateBuffer.get(),
    TONEMAP_STATE_MAX_OFFSET,
    TONEMAP_STATE_CLEAR_END - TONEMAP_STATE_MAX_OFFSET,
    0u);

  memory_barrier(
    cmd_buf,
    vk::PipelineStageFlagBits2::eAllTransfer,
    vk::AccessFlagBits2::eTransferWrite,
    COMPUTE,
    STORAGE_RW);

  const auto runPass =
    [&](const char* program, const etna::ComputePipeline& pipeline, std::string_view section) {
      auto info = etna::get_shader_program(program);
      auto set = etna::create_descriptor_set(
        info.getDescriptorLayoutId(0),
        cmd_buf,
        {
          etna::Binding{
            0, hdr.genBinding(pointSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
          etna::Binding{1, stateBuffer.genBinding()},
        });
      etna::flush_barriers(cmd_buf);

      timer.begin(cmd_buf, section);
      cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.getVkPipeline());
      cmd_buf.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute, pipeline.getVkPipelineLayout(), 0, {set.getVkSet()}, {});
      cmd_buf.pushConstants<TonemapPush>(
        pipeline.getVkPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0, {push});
      cmd_buf.dispatch(groups.x, groups.y, 1);
      timer.end(cmd_buf);

      memory_barrier(cmd_buf, COMPUTE, STORAGE_RW, COMPUTE, STORAGE_RW);
    };

  runPass(
    useSharedMemory ? "lum_minmax_shared" : "lum_minmax_global",
    minmaxPipelines[variant],
    useSharedMemory ? "Lum min/max (shared)" : "Lum min/max (global)");
  runPass(
    useSharedMemory ? "lum_histogram_shared" : "lum_histogram_global",
    histogramPipelines[variant],
    useSharedMemory ? "Histogram (shared)" : "Histogram (global)");

  {
    auto info = etna::get_shader_program("exposure");
    auto set = etna::create_descriptor_set(
      info.getDescriptorLayoutId(0), cmd_buf, {etna::Binding{0, stateBuffer.genBinding()}});
    timer.begin(cmd_buf, "Exposure");
    cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, exposurePipeline.getVkPipeline());
    cmd_buf.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,
      exposurePipeline.getVkPipelineLayout(),
      0,
      {set.getVkSet()},
      {});
    cmd_buf.pushConstants<TonemapPush>(
      exposurePipeline.getVkPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0, {push});
    cmd_buf.dispatch(1, 1, 1);
    timer.end(cmd_buf);
  }

  // The exposure is read by the post-processing pass and copied back for the GUI.
  // The copy also has to be ordered after the copy into the same readback buffer made by the
  // frame that used this slot before (write-after-write across submissions).
  memory_barrier(
    cmd_buf,
    COMPUTE | vk::PipelineStageFlagBits2::eCopy,
    vk::AccessFlagBits2::eShaderStorageWrite | vk::AccessFlagBits2::eTransferWrite,
    vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eCopy,
    vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eTransferRead |
      vk::AccessFlagBits2::eTransferWrite);

  cmd_buf.copyBuffer(
    stateBuffer.get(), readbackBuffers->get().get(), {vk::BufferCopy{0, 0, sizeof(TonemapState)}});
  memory_barrier(
    cmd_buf,
    vk::PipelineStageFlagBits2::eCopy,
    vk::AccessFlagBits2::eTransferWrite,
    vk::PipelineStageFlagBits2::eHost,
    vk::AccessFlagBits2::eHostRead);
}

void Tonemapper::postProcess(
  vk::CommandBuffer cmd_buf,
  const etna::Image& hdr,
  vk::Image target_image,
  vk::ImageView target_view,
  glm::uvec2 resolution,
  bool debug_view,
  GpuTimer& timer)
{
  ETNA_PROFILE_GPU(cmd_buf, postProcess);

  auto info = etna::get_shader_program("postprocess");
  auto set = etna::create_descriptor_set(
    info.getDescriptorLayoutId(0),
    cmd_buf,
    {
      etna::Binding{0, hdr.genBinding(pointSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
      etna::Binding{1, stateBuffer.genBinding()},
      etna::Binding{
        2, lutImage.genBinding(lutSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
    });

  const PostPush push{
    .flags = glm::vec4(
      tonemapEnabled ? 1.0f : 0.0f,
      static_cast<float>(tonemapOperator),
      srgbEncoding ? 1.0f : 0.0f,
      colorGrading ? 1.0f : 0.0f),
    .settings =
      glm::vec4(manualExposure, lutStrength, dithering ? 1.0f : 0.0f, debug_view ? 1.0f : 0.0f),
  };

  timer.begin(cmd_buf, "Post-process");
  {
    etna::RenderTargetState renderTargets(
      cmd_buf,
      {{0, 0}, {resolution.x, resolution.y}},
      {{.image = target_image, .view = target_view, .loadOp = vk::AttachmentLoadOp::eDontCare}},
      {});
    cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, postPipeline.getVkPipeline());
    cmd_buf.bindDescriptorSets(
      vk::PipelineBindPoint::eGraphics,
      postPipeline.getVkPipelineLayout(),
      0,
      {set.getVkSet()},
      {});
    cmd_buf.pushConstants<PostPush>(
      postPipeline.getVkPipelineLayout(), vk::ShaderStageFlagBits::eFragment, 0, {push});
    cmd_buf.draw(3, 1, 0, 0);
  }
  timer.end(cmd_buf);
}

namespace
{

float decode_ordered(std::uint32_t u)
{
  const std::uint32_t bits = (u & 0x80000000u) != 0u ? (u & 0x7FFFFFFFu) : ~u;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

} // namespace

void Tonemapper::printStatistics() const
{
  std::uint64_t counted = 0;
  for (std::uint32_t c : stats.histogram)
    counted += c;
  spdlog::info(
    "Tone mapping: {}, log2 lum [{:.2f}, {:.2f}], target {:.2f}, adapted {:.2f}, exposure {:.3f}, "
    "{} pixels in the histogram",
    useSharedMemory ? "shared memory" : "global atomics",
    decode_ordered(stats.minLogBits),
    decode_ordered(stats.maxLogBits),
    stats.targetLogLum,
    stats.adaptedLogLum,
    stats.exposure,
    counted);
}

void Tonemapper::drawGui(const GpuTimer& timer)
{
  ImGui::Checkbox("Tone mapping (auto exposure + curve)", &tonemapEnabled);
  ImGui::SameLine();
  ImGui::Checkbox("Color grading (3D LUT)", &colorGrading);
  ImGui::Checkbox("sRGB encoding (gamma)", &srgbEncoding);
  ImGui::SameLine();
  ImGui::Checkbox("Dithering", &dithering);

  ImGui::SeparatorText("Tone mapping");
  const char* operators[] = {
    "Clamp (naive)", "Reinhard (luminance)", "ACES (Narkowicz)", "Hable (Uncharted 2)"};
  ImGui::Combo("Curve", &tonemapOperator, operators, IM_ARRAYSIZE(operators));
  if (!tonemapEnabled)
    ImGui::SliderFloat(
      "Manual exposure", &manualExposure, 0.05f, 8.0f, "%.2f", ImGuiSliderFlags_Logarithmic);

  const char* methods[] = {"Trimmed mean of log lum", "Median", "Mean of log lum"};
  ImGui::Combo("Exposure from", &exposureMethod, methods, IM_ARRAYSIZE(methods));
  if (exposureMethod == EXPOSURE_TRIMMED_MEAN)
    ImGui::DragFloatRange2(
      "Percentiles", &lowPercentile, &highPercentile, 0.005f, 0.0f, 1.0f, "%.3f");
  ImGui::SliderFloat("Key value (middle grey)", &keyValue, 0.02f, 0.8f);
  ImGui::SliderFloat("Exposure compensation (EV)", &exposureCompensation, -4.0f, 4.0f);
  ImGui::DragFloatRange2("Allowed log2 lum", &minLogLum, &maxLogLum, 0.1f, -16.0f, 16.0f, "%.1f");
  ImGui::SliderFloat(
    "Adaptation to light (1/s)", &adaptSpeedUp, 0.1f, 20.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
  ImGui::SliderFloat(
    "Adaptation to dark (1/s)", &adaptSpeedDown, 0.1f, 20.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
  ImGui::Checkbox("Shared memory in min/max + histogram", &useSharedMemory);

  // Statistics read back from the GPU
  const float lo = decode_ordered(stats.minLogBits);
  const float hi = decode_ordered(stats.maxLogBits);
  std::uint64_t counted = 0;
  for (std::uint32_t c : stats.histogram)
    counted += c;
  ImGui::Text("log2 luminance: min %.2f, max %.2f (%.1f stops)", lo, hi, hi - lo);
  ImGui::Text("pixels in the histogram: %llu", static_cast<unsigned long long>(counted));
  ImGui::Text(
    "target %.2f, adapted %.2f, exposure x%.3f",
    stats.targetLogLum,
    stats.adaptedLogLum,
    stats.exposure);
  ImGui::PlotHistogram(
    "##lum_hist",
    histogramPlot.data(),
    HIST_BINS,
    0,
    "log2 luminance histogram",
    0.0f,
    FLT_MAX,
    ImVec2(0, 70));

  for (const char* name :
       {"Lum min/max (shared)",
        "Lum min/max (global)",
        "Histogram (shared)",
        "Histogram (global)",
        "Exposure",
        "Post-process"})
    if (const auto* s = timer.find(name))
      ImGui::Text("%-22s %7.3f ms", name, s->smoothedMs);

  ImGui::SeparatorText("Color grading");
  if (ImGui::BeginCombo(
        "LUT", luts.empty() ? "" : luts[static_cast<std::size_t>(lutIndex)].name.c_str()))
  {
    for (std::size_t i = 0; i < luts.size(); ++i)
      if (ImGui::Selectable(luts[i].name.c_str(), static_cast<std::size_t>(lutIndex) == i))
        lutIndex = static_cast<int>(i);
    ImGui::EndCombo();
  }
  ImGui::SliderFloat("LUT strength", &lutStrength, 0.0f, 1.0f);
  if (!luts.empty() && luts[static_cast<std::size_t>(lutIndex)].filledCells > 0)
    ImGui::Text(
      "%u / %u cells filled from the images, the rest from the nearest",
      luts[static_cast<std::size_t>(lutIndex)].filledCells,
      lut::SIZE * lut::SIZE * lut::SIZE);
}
