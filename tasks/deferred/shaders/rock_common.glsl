#ifndef ROCK_COMMON_GLSL_INCLUDED
#define ROCK_COMMON_GLSL_INCLUDED

#include "SceneParams.h"

layout(binding = 1) uniform SceneUbo
{
  SceneParams params;
};

layout(push_constant) uniform RockPush
{
  vec4 rock; // x = first instance of this model in the buffer, y = roughness, z = metallic
} pc;

#endif // ROCK_COMMON_GLSL_INCLUDED
