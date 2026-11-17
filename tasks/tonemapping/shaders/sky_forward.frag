#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

// Sky background of the forward reference path: the same cubemap lookup as in the deferred
// lighting pass, drawn before the geometry.

#include "SceneParams.h"

layout(binding = 0) uniform SceneUbo
{
  SceneParams params;
};

layout(binding = 1) uniform samplerCube skyCube;

#include "shading.glsl"

layout(location = 0) in vec2 vUv;

layout(location = 0) out vec4 out_fragColor;

void main()
{
  const vec2 uv = gl_FragCoord.xy * params.screen.zw;
  const vec4 rel = params.invViewProjRel * vec4(uv * 2.0 - 1.0, 1.0, 1.0);
  const vec3 dir = normalize(rel.xyz / rel.w);
  const int mode = int(params.debug.x + 0.5);
  out_fragColor = vec4(mode == DEBUG_VIEW_SHADED ? skyRadiance(dir) : vec3(0.0), 1.0);
}
