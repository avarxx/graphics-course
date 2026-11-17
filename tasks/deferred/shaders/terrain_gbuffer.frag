#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

// Terrain G-buffer pass: no lighting here, only the surface parameters are written out.

#include "terrain_inputs.glsl"
#include "terrain_surface.glsl"
#include "gbuffer.glsl"

layout(location = 0) out vec4 out_albedo;
layout(location = 1) out vec4 out_normal;
layout(location = 2) out vec4 out_material;

void main()
{
  const Surface s = evalTerrainSurface(surf.wPos, surf.uv);
  const GBufferTexel t = packGBuffer(s, params.gbuffer.x > 0.5);
  out_albedo = t.rt0;
  out_normal = t.rt1;
  out_material = t.rt2;
}
