#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

// Depth-only pass for the sun shadow map. Reuses the terrain height sampling of the main pass,
// but projects with the light matrix instead of the camera matrix.

#include "SceneParams.h"
#include "terrain_common.glsl"

layout(location = 0) in vec2 vGridUv;

layout(binding = 0) uniform SceneUbo
{
  SceneParams params;
};

layout(binding = 1) uniform sampler2D heightMap;

out gl_PerVertex { vec4 gl_Position; };

void main(void)
{
  const float worldSize = params.terrainSize.x;
  const float heightScale = params.terrainSize.y;

  const float h = sampleHeight(heightMap, vGridUv);
  const vec2 xz = uvToWorldXZ(vGridUv, worldSize);

  gl_Position = params.shadowViewProj * vec4(xz.x, h * heightScale, xz.y, 1.0);
}
