#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "rock_surface.glsl"

layout(binding = 2) uniform samplerCube skyCube;

#include "shading.glsl"

layout(location = 0) out vec4 out_fragColor;

void main()
{
  out_fragColor = vec4(shadeSurface(evalRockSurface(), surf.wPos), 1.0);
}
