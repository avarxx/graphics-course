#pragma once

#include <array>
#include <optional>

#include <etna/Image.hpp>
#include <etna/Sampler.hpp>
#include <etna/Buffer.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <etna/GpuSharedResource.hpp>
#include <etna/OneShotCmdMgr.hpp>
#include <glm/glm.hpp>

#include "wsi/Keyboard.hpp"
#include "shaders/SceneParams.h"

#include "Decals.hpp"
#include "FramePacket.hpp"
#include "GBuffer.hpp"
#include "GpuTimer.hpp"
#include "HeightmapGenerator.hpp"
#include "RockModel.hpp"


class WorldRenderer
{
public:
  WorldRenderer();

  void loadShaders();
  void allocateResources(glm::uvec2 swapchain_resolution);
  void loadResources();
  void setupPipelines(vk::Format swapchain_format);

  void debugInput(const Keyboard& kb);
  void update(const FramePacket& packet);
  void drawGui();
  void renderWorld(
    vk::CommandBuffer cmd_buf, vk::Image target_image, vk::ImageView target_image_view);

  // Settings that can be overridden from the command line (used for automated screenshots)
  struct Overrides
  {
    std::optional<bool> deferredMode;
    std::optional<GBufferLayout> layout;
    std::optional<bool> decals;
    std::optional<int> debugView;
  };
  void applyOverrides(const Overrides& o);
  void printTimings() const;

private:
  // --- Passes ---------------------------------------------------------------------------------
  void renderDeferred(vk::CommandBuffer cmd_buf, vk::Image target_image, vk::ImageView target_view);
  void renderForward(vk::CommandBuffer cmd_buf, vk::Image target_image, vk::ImageView target_view);
  void depthPrepass(vk::CommandBuffer cmd_buf, GBuffer& gb);
  void geometryPass(vk::CommandBuffer cmd_buf, GBuffer& gb);
  void decalPass(vk::CommandBuffer cmd_buf, GBuffer& gb);
  void lightingPass(
    vk::CommandBuffer cmd_buf, GBuffer& gb, vk::Image target_image, vk::ImageView target_view);

  // Scene drawing with an already bound pipeline
  enum class TerrainProgram
  {
    DepthOnly,
    GBuffer,
    Forward,
  };
  void drawTerrain(
    vk::CommandBuffer cmd_buf, const etna::GraphicsPipeline& pipeline, TerrainProgram program);
  void drawRocks(
    vk::CommandBuffer cmd_buf, const etna::GraphicsPipeline& pipeline, TerrainProgram program);

  // --- Scene ----------------------------------------------------------------------------------
  void regenerateHeightmap();
  void createTerrainGrid();
  void loadTerrainTextures();
  glm::vec3 sunDirection() const;
  void loadRocks();
  void gatherRockInstances(const Camera& cam);
  TerrainInfo terrainInfo() const;

  struct RockInstance
  {
    glm::mat4 model;
    glm::vec4 color;
  };

  struct RockPushConstants
  {
    glm::vec4 rock; // x = first instance, y = roughness, z = metallic
  };

  struct DecalPushConstants
  {
    glm::vec4 global; // x = global opacity, y = show boxes, z = compact G-buffer
  };

  static constexpr std::uint32_t ROCK_MODELS = 2;
  static constexpr std::uint32_t MAX_ROCKS_PER_MODEL = 32768;

  std::array<RockModel, ROCK_MODELS> rockModels;
  // Per-frame-in-flight host-visible instance buffers: CPU writes frame N+1 while GPU reads frame N
  std::optional<etna::GpuSharedResource<etna::Buffer>> rockInstanceBuffer;
  std::array<std::uint32_t, ROCK_MODELS> visibleRocks{};
  std::vector<RockInstance> rockScratch[ROCK_MODELS];

  bool drawRocksEnabled = true;
  float rockDensity = 0.5f;  // probability of a rock per cell
  float rockCellSize = 6.0f; // meters
  float rockDrawDistance = 250.0f;
  glm::vec2 rockScaleRange{0.4f, 2.2f};
  float rockRoughness = 0.85f;
  float rockMetallic = 0.0f;
  bool frustumCulling = true;
  std::uint32_t rocksConsidered = 0;
  float rockCpuMs = 0.0f;

  static constexpr std::uint32_t GRID_CELLS = 256;

  std::unique_ptr<etna::OneShotCmdMgr> oneShotCmdMgr;

  // --- Deferred shading -----------------------------------------------------------------------
  // Both layouts are allocated up front, so switching between them in ImGui is instant
  std::array<GBuffer, 2> gbuffers;
  GBufferLayout gbufferLayout = GBufferLayout::Compact;
  bool deferredMode = true; // false = forward reference renderer (the previous task)
  bool useDepthPrepass = true;
  GBuffer& activeGBuffer() { return gbuffers[static_cast<std::size_t>(gbufferLayout)]; }

  etna::Sampler pointSampler;
  DecalSystem decals;
  GpuTimer gpuTimer;

  // Heightmap: CPU copy is kept for gameplay-ish queries (e.g. object placement)
  HeightmapParams heightmapParams;
  std::vector<float> heights;
  etna::Image heightMap;
  bool heightmapDirty = true;
  bool autoRegenerate = false;
  float lastGenerationMs = 0.0f;

  // Terrain materials: 4 layers (sand, grass, rock, snow)
  etna::Image albedoArray;
  etna::Image roughHeightArray;
  int loadedTextureLayers = 0;

  etna::Sampler heightSampler;
  etna::Sampler materialSampler;
  etna::Sampler decalSampler;

  etna::Buffer gridVertices;
  etna::Buffer gridIndices;
  std::uint32_t gridIndexCount = 0;

  // One uniform buffer per frame in flight, so that we never overwrite data in use by the GPU
  std::optional<etna::GpuSharedResource<etna::Buffer>> sceneUbo;
  SceneParams sceneParams{};

  // GUI-controlled settings
  float worldSize = 600.0f;
  float heightScale = 110.0f;
  float seaLevel = 0.1f;
  glm::vec4 layerHeights{0.13f, 0.45f, 0.74f, 0.035f};
  float textureTiling = 6.0f;
  float detailStrength = 1.0f;
  float slopeRock = 1.0f;
  float heightBlendSharpness = 4.0f;
  float roughnessScale = 1.0f;
  float cavityAo = 0.0f;
  bool detailNormals = true;
  int debugView = 0;

  float sunAzimuth = 40.0f;
  float sunElevation = 32.0f;
  float sunIntensity = 3.2f;
  glm::vec3 sunColor{1.0f, 0.95f, 0.85f};
  float ambientIntensity = 0.45f;
  float ambientSpecular = 1.0f;
  glm::vec3 skyColor{0.45f, 0.62f, 0.85f};
  glm::vec3 groundAmbient{0.25f, 0.22f, 0.18f};
  float exposure = 1.0f;
  float fogDensity = 0.0008f;

  glm::mat4x4 worldViewProj{1.0f};
  glm::vec3 cameraPos{0.0f};
  glm::vec3 cameraForward{0.0f, 0.0f, -1.0f};
  float currentTime = 0.0f;

  // --- Pipelines ------------------------------------------------------------------------------
  etna::GraphicsPipeline terrainDepthPipeline{};
  etna::GraphicsPipeline rockDepthPipeline{};
  // [layout][with depth prepass ? EQUAL test : LESS test + depth write]
  etna::GraphicsPipeline terrainGBufferPipelines[2][2]{};
  etna::GraphicsPipeline rockGBufferPipelines[2][2]{};
  etna::GraphicsPipeline decalPipelines[2]{};
  etna::GraphicsPipeline lightingPipeline{};
  etna::GraphicsPipeline terrainForwardPipeline{};
  etna::GraphicsPipeline rockForwardPipeline{};

  glm::uvec2 resolution{};
};
