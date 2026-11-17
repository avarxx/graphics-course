#ifndef TERRAIN_INPUTS_GLSL_INCLUDED
#define TERRAIN_INPUTS_GLSL_INCLUDED

#include "SceneParams.h"

layout(location = 0) in VS_OUT
{
  vec3 wPos;
  vec2 uv;
} surf;

layout(binding = 0) uniform SceneUbo
{
  SceneParams params;
};

layout(binding = 1) uniform sampler2D heightMap;
// Layers: 0 = sand, 1 = grass, 2 = rock, 3 = snow
layout(binding = 2) uniform sampler2DArray albedoArray;
// R = roughness, G = height (displacement)
layout(binding = 3) uniform sampler2DArray roughHeightArray;

#endif // TERRAIN_INPUTS_GLSL_INCLUDED
