#pragma once

#include <optional>
#include <vector>

#include <etna/Buffer.hpp>
#include <etna/GpuSharedResource.hpp>
#include <etna/Image.hpp>
#include <etna/OneShotCmdMgr.hpp>
#include <glm/glm.hpp>

#include "shaders/DecalData.h"


enum class DecalKind : int
{
  Puddle = 0,
  CrackedMud,
  Moss,
  Scorch,
  PaintMark,
  Count,
};

// One projective decal as edited in ImGui. The box stands on the terrain: its center is placed
// at the terrain height under (x, z), and it projects its textures straight down (local -Y).
struct Decal
{
  bool enabled = true;
  DecalKind kind = DecalKind::Puddle;
  glm::vec2 xz{0.0f};
  float yaw = 0.0f;      // degrees
  glm::vec2 size{20.0f}; // footprint, meters
  float depth = 24.0f;   // box height, meters
  float opacity = 1.0f;
  float albedoIntensity = 1.0f;
  float normalIntensity = 1.0f;
  float roughnessIntensity = 1.0f;
  glm::vec3 tint{1.0f};
};

struct TerrainInfo
{
  const std::vector<float>* heights = nullptr;
  float worldSize = 600.0f;
  float heightScale = 110.0f;
  float seaLevel = 0.1f;
};

class DecalSystem
{
public:
  void load(etna::OneShotCmdMgr& cmd_mgr);

  // Builds the GPU data for this frame (into the per-frame-in-flight buffer)
  void update(const TerrainInfo& terrain);

  void drawGui(const TerrainInfo& terrain, glm::vec3 camera_pos, glm::vec3 camera_forward);

  std::uint32_t gpuCount() const { return activeCount; }
  const etna::Buffer& currentBuffer() const { return buffer->get(); }

  bool enabled = true;
  float globalOpacity = 1.0f;
  bool showBoxes = false;
  float angleFade = 0.35f; // cos of the angle at which decals start fading on slopes

  etna::Image albedoArray; // rgb = albedo (sRGB), a = coverage mask
  etna::Image normalArray; // rgb = tangent space normal, a = roughness

private:
  void placeDefaults();
  void scatterRandom(const TerrainInfo& terrain, int count, std::uint32_t seed);

  std::vector<Decal> decals;
  std::optional<etna::GpuSharedResource<etna::Buffer>> buffer;
  std::uint32_t activeCount = 0;
  int scatterCount = 24;
  int scatterSeed = 7;
  bool snapPending = false;
};
