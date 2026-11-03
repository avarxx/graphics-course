#pragma once

#include <filesystem>
#include <vector>

#include <etna/Buffer.hpp>
#include <etna/OneShotCmdMgr.hpp>
#include <glm/glm.hpp>


struct RockVertex
{
  glm::vec3 position;
  glm::vec3 normal;
};

// Geometry of one rock model in its own vertex and index buffers
struct RockModel
{
  etna::Buffer vertexBuffer;
  etna::Buffer indexBuffer;
  std::uint32_t indexCount = 0;
  float boundingRadius = 1.0f;
  glm::vec3 baseColor{0.5f};

  // Loads all primitives of a glTF file (node transforms are baked in) and normalizes
  // the model to unit size. Falls back to a procedural rock if the file is missing.
  void load(etna::OneShotCmdMgr& cmd_mgr, const std::filesystem::path& path, std::uint32_t seed);
};
