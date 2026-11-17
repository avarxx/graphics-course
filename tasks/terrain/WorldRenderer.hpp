#pragma once

#include <optional>

#include <etna/Image.hpp>
#include <etna/Sampler.hpp>
#include <etna/Buffer.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <etna/GpuSharedResource.hpp>
#include <etna/OneShotCmdMgr.hpp>
#include <glm/glm.hpp>

#include "wsi/Keyboard.hpp"
#include "shaders/TerrainParams.h"

#include "FramePacket.hpp"
#include "HeightmapGenerator.hpp"


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

private:
  void regenerateHeightmap();
  void createTerrainGrid();
  void loadTerrainTextures();
  void renderTerrain(vk::CommandBuffer cmd_buf);
  glm::vec3 sunDirection() const;

private:
  static constexpr std::uint32_t GRID_CELLS = 256;

  std::unique_ptr<etna::OneShotCmdMgr> oneShotCmdMgr;

  etna::Image mainViewDepth;

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

  etna::Buffer gridVertices;
  etna::Buffer gridIndices;
  std::uint32_t gridIndexCount = 0;

  // One uniform buffer per frame in flight, so that we never overwrite data in use by the GPU
  std::optional<etna::GpuSharedResource<etna::Buffer>> terrainUbo;
  TerrainParams terrainParams{};

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
  bool detailNormals = true;
  int debugView = 0;

  float sunAzimuth = 40.0f;
  float sunElevation = 32.0f;
  float sunIntensity = 3.2f;
  glm::vec3 sunColor{1.0f, 0.95f, 0.85f};
  float ambientIntensity = 0.45f;
  glm::vec3 skyColor{0.45f, 0.62f, 0.85f};
  glm::vec3 groundAmbient{0.25f, 0.22f, 0.18f};
  float exposure = 1.0f;
  float fogDensity = 0.0008f;

  glm::mat4x4 worldViewProj{1.0f};
  glm::vec3 cameraPos{0.0f};
  float currentTime = 0.0f;

  etna::GraphicsPipeline terrainPipeline{};

  glm::uvec2 resolution{};
};
