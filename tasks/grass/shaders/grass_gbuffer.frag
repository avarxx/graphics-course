#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

// Grass writes the same Surface parameters as terrain and rocks, so the deferred lighting pass
// shades it with the exact same PBR code. Grass is a thin double-sided ribbon: the normal is
// flipped towards the camera and the base is darkened (ambient occlusion towards the ground).

#include "SceneParams.h"
#include "gbuffer.glsl"

layout(binding = 1) uniform SceneUbo
{
  SceneParams params;
};

layout(location = 0) in VS_OUT
{
  vec3 wPos;
  vec3 wNorm;
  vec3 color;
  float t;
} surf;

layout(location = 0) out vec4 out_albedo;
layout(location = 1) out vec4 out_normal;
layout(location = 2) out vec4 out_material;

void main()
{
  const vec3 V = normalize(params.cameraPos.xyz - surf.wPos);
  vec3 N = normalize(surf.wNorm);
  if (dot(N, V) < 0.0)
    N = -N;

  Surface s;
  s.albedo = surf.color;
  s.normal = N;
  s.roughness = 0.62;
  s.metallic = 0.0;
  s.ao = mix(0.35, 1.0, surf.t);

  const GBufferTexel t = packGBuffer(s, params.gbuffer.x > 0.5);
  out_albedo = t.rt0;
  out_normal = t.rt1;
  out_material = t.rt2;
}
