#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "pbr.glsl"

layout(location = 0) out vec4 out_fragColor;

layout(location = 0) in VS_OUT
{
  vec3 wPos;
  vec3 wNorm;
  vec3 color;
} surf;

layout(push_constant) uniform Params
{
  mat4 viewProj;
  vec4 cameraPos;
  vec4 sunDir;
  vec4 sunColor;
} params;

void main()
{
  const vec3 N = normalize(surf.wNorm);
  const vec3 V = normalize(params.cameraPos.xyz - surf.wPos);
  const vec3 L = normalize(params.sunDir.xyz);
  const vec3 albedo = surf.color;
  const float roughness = 0.85;

  vec3 color = shadeDirectional(albedo, roughness, 0.0, N, V, L, params.sunColor.rgb * params.sunDir.w);
  color += shadeAmbient(albedo, 1.0, N, vec3(0.45, 0.62, 0.85), vec3(0.25, 0.22, 0.18), params.sunColor.w);
  out_fragColor = vec4(tonemapACES(color), 1.0);
}
