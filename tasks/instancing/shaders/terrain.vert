#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "TerrainParams.h"
#include "terrain_common.glsl"


layout(location = 0) in vec2 vGridUv;

layout(binding = 0) uniform TerrainUbo
{
  TerrainParams params;
};

layout(binding = 1) uniform sampler2D heightMap;

layout(location = 0) out VS_OUT
{
  vec3 wPos;
  vec2 uv;
} vOut;

out gl_PerVertex { vec4 gl_Position; };

void main(void)
{
  const float worldSize = params.terrainSize.x;
  const float heightScale = params.terrainSize.y;

  const float h = sampleHeight(heightMap, vGridUv);
  const vec2 xz = uvToWorldXZ(vGridUv, worldSize);

  vOut.wPos = vec3(xz.x, h * heightScale, xz.y);
  vOut.uv = vGridUv;

  gl_Position = params.viewProj * vec4(vOut.wPos, 1.0);
}
