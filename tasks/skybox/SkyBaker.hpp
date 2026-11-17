#pragma once

#include <optional>

#include <etna/Buffer.hpp>
#include <etna/ComputePipeline.hpp>
#include <etna/GpuSharedResource.hpp>
#include <etna/Image.hpp>
#include <etna/Sampler.hpp>
#include <glm/glm.hpp>

#include "shaders/SkyParams.h"

class GpuTimer;


/**
 * Procedural sky with volumetric clouds, baked into a cubemap by a compute shader.
 *
 * The cubemap is re-baked only when the parameters change (or on request), otherwise the
 * baked texture is reused. After baking, a mip chain is generated with blits: the lighting pass
 * uses the blurry mips for ambient diffuse / specular lighting and for the fog color.
 */
class SkyBaker
{
public:
  static constexpr std::uint32_t DEFAULT_FACE_SIZE = 2048;
  static constexpr vk::Format FORMAT = vk::Format::eR16G16B16A16Sfloat;

  // face_size: 2048 by the task, smaller sizes are only for quick tests (--sky-size)
  void init(std::uint32_t face_size = DEFAULT_FACE_SIZE);
  void loadShaders(const char* shaders_root);
  void setupPipelines();

  // Sun parameters are owned by the world renderer (they also drive the terrain lighting)
  void setSun(glm::vec3 direction, glm::vec3 color, float intensity);

  // Records the bake (if anything changed) into the frame's command buffer.
  // After this call the cubemap is ready to be sampled.
  void bakeIfNeeded(vk::CommandBuffer cmd_buf, GpuTimer& timer);

  // Returns true if anything sky-related was changed
  bool drawGui();

  const etna::Image& cubemap() const { return cube; }
  vk::Sampler sampler() const { return cubeSampler.get(); }
  std::uint32_t mipCount() const { return mips; }

  // Number of complete bakes so far (for the GUI / logs)
  std::uint32_t bakeCount() const { return bakes; }
  void requestBake() { dirty = true; }

private:
  void bakeFaces(vk::CommandBuffer cmd_buf, std::uint32_t first_face, std::uint32_t face_count);
  void generateMips(vk::CommandBuffer cmd_buf);
  void updateParams();

  etna::Image cube;
  etna::Sampler cubeSampler;
  std::optional<etna::GpuSharedResource<etna::Buffer>> paramsUbo;
  etna::ComputePipeline bakePipeline{};
  std::uint32_t mips = 1;
  std::uint32_t faceSize = DEFAULT_FACE_SIZE;

  SkyParams params{};
  bool dirty = true;
  // Progressive baking: how many faces are baked per frame (6 = whole cubemap at once)
  int facesPerFrame = 6;
  std::uint32_t nextFace = 0;
  std::uint32_t bakes = 0;

public:
  // --- GUI-controlled settings ---------------------------------------------------------------
  glm::vec3 horizonColor{0.60f, 0.78f, 0.98f}; // light blue
  glm::vec3 zenithColor{0.10f, 0.24f, 0.58f};  // dark blue
  float skyBrightness = 1.0f;
  float glowNarrow = 1.0f;
  float glowWide = 0.35f;
  float glowNarrowSharpness = 9000.0f;
  float glowWideFalloff = 42.0f;

  bool cloudsEnabled = true;
  float coverage = 0.5f;
  float density = 0.035f;     // extinction per meter at density 1
  float cloudScale = 3500.0f; // meters per noise period
  float layerBottom = 1500.0f;
  float layerTop = 4000.0f;
  int primarySteps = 64;
  int lightSteps = 6;
  float sunScattering = 2.0f;
  float cloudAmbient = 1.0f;
  float forwardScattering = 0.6f;
  float erosion = 0.35f;
  float hazeDistance = 60000.0f;
  glm::vec3 windOffset{0.0f};

private:
  glm::vec3 sunDir{0, 1, 0};
  glm::vec3 sunColor{1.0f};
  float sunIntensity = 1.0f;
};
