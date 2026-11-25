#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

#include <etna/Buffer.hpp>
#include <etna/ComputePipeline.hpp>
#include <etna/GpuSharedResource.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <etna/Image.hpp>
#include <etna/OneShotCmdMgr.hpp>
#include <etna/Sampler.hpp>
#include <glm/glm.hpp>

#include "shaders/TonemapParams.h"

class GpuTimer;


/**
 * HDR -> LDR: automatic exposure from a luminance histogram (compute shaders), tone mapping
 * curve, sRGB encoding and 3D LUT color grading (post-processing fragment shader).
 *
 *   minmax_*.comp      min / max log2 luminance of the HDR image
 *   histogram_*.comp   HIST_BINS-bin histogram of log2 luminance between min and max
 *   exposure.comp      exposure from the histogram + temporal adaptation (GPU only, no readback)
 *   postprocess.frag   exposure, curve, sRGB, LUT, dithering -> backbuffer
 *
 * Every pass over the image exists in two variants (shared memory / global atomics only) to
 * compare them in the profiler.
 */
class Tonemapper
{
public:
  // The HDR render target format required by the task
  static constexpr vk::Format HDR_FORMAT = vk::Format::eB10G11R11UfloatPack32;

  void init(etna::OneShotCmdMgr& cmd_mgr);
  void loadShaders(const char* shaders_root);
  void setupPipelines(vk::Format swapchain_format);

  // Called before recording the frame: uploads a newly selected LUT, reads the statistics
  void update();

  // Records the min/max, histogram and exposure passes. The HDR image must be fully rendered.
  void computeExposure(
    vk::CommandBuffer cmd_buf,
    const etna::Image& hdr,
    glm::uvec2 resolution,
    float dt,
    GpuTimer& timer);

  // Draws a fullscreen triangle from the HDR image into the backbuffer
  void postProcess(
    vk::CommandBuffer cmd_buf,
    const etna::Image& hdr,
    vk::Image target_image,
    vk::ImageView target_view,
    glm::uvec2 resolution,
    bool debug_view,
    GpuTimer& timer);

  void drawGui(const GpuTimer& timer);
  void printStatistics() const;

  // --- Settings (ImGui / command line) -------------------------------------------------------
  bool tonemapEnabled = true; // auto exposure + curve; off = plain clamp (step 1)
  bool colorGrading = true;
  bool srgbEncoding = true;
  bool dithering = true;
  bool useSharedMemory = true;
  int tonemapOperator = TONEMAP_ACES;
  int exposureMethod = EXPOSURE_TRIMMED_MEAN;
  float lowPercentile = 0.40f;
  float highPercentile = 0.95f;
  float keyValue = 0.18f;
  float exposureCompensation = 0.0f; // EV
  float minLogLum = -8.0f;
  float maxLogLum = 10.0f;
  float adaptSpeedUp = 3.0f;   // to brighter scenes
  float adaptSpeedDown = 1.2f; // to darker scenes
  float manualExposure = 1.0f; // when tone mapping is off
  float lutStrength = 1.0f;
  int lutIndex = 0;

  // Selects a LUT by its folder name, returns false if there is no such LUT
  bool selectLut(const std::string& name);

private:
  void uploadLut(etna::OneShotCmdMgr& cmd_mgr, std::size_t index);
  TonemapPush makePush(glm::uvec2 resolution, float dt) const;
  // Reads the statistics copied back by an earlier frame (for ImGui only, nothing depends on it)
  void readStatistics();

  etna::Buffer stateBuffer;
  std::optional<etna::GpuSharedResource<etna::Buffer>> readbackBuffers;
  etna::Image lutImage;
  etna::Sampler pointSampler;
  etna::Sampler lutSampler;

  struct LutEntry
  {
    std::string name;
    std::vector<glm::u8vec4> texels;
    std::uint32_t filledCells = 0;
  };
  std::vector<LutEntry> luts;
  std::size_t uploadedLut = ~std::size_t{0};
  etna::OneShotCmdMgr* cmdMgr = nullptr;

  etna::ComputePipeline minmaxPipelines[2]{};    // [global, shared]
  etna::ComputePipeline histogramPipelines[2]{}; // [global, shared]
  etna::ComputePipeline exposurePipeline{};
  etna::GraphicsPipeline postPipeline{};

  // Last statistics read back from the GPU (a couple of frames old)
  TonemapState stats{};
  std::array<float, HIST_BINS> histogramPlot{};
};
