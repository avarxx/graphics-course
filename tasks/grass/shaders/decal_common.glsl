#ifndef DECAL_COMMON_GLSL_INCLUDED
#define DECAL_COMMON_GLSL_INCLUDED

#include "SceneParams.h"
#include "DecalData.h"

layout(binding = 0) uniform SceneUbo
{
  SceneParams params;
};

layout(std430, binding = 1) readonly buffer Decals
{
  DecalGpu decals[];
};

layout(push_constant) uniform DecalPush
{
  vec4 global; // x = global opacity, y = show decal boxes, z = compact G-buffer
} pc;

#endif // DECAL_COMMON_GLSL_INCLUDED
