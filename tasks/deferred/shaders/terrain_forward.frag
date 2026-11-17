#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

// Forward reference path (the renderer of the previous task), kept to check that deferred
// shading produces the same image. Shares the material and lighting code with the deferred path.

#include "terrain_inputs.glsl"
#include "terrain_surface.glsl"
#include "shading.glsl"

layout(location = 0) out vec4 out_fragColor;

void main()
{
  const Surface s = evalTerrainSurface(surf.wPos, surf.uv);
  out_fragColor = vec4(shadeSurface(s, surf.wPos), 1.0);
}
