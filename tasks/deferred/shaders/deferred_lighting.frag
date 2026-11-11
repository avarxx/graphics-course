#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

// Deferred lighting (G-buffer resolve): one fullscreen triangle, every pixel reads its surface
// from the G-buffer, restores the world position from depth and runs the Cook-Torrance BRDF.

#include "SceneParams.h"

layout(binding = 0) uniform SceneUbo
{
  SceneParams params;
};

#include "gbuffer.glsl"
#include "shading.glsl"

layout(binding = 1) uniform sampler2D gAlbedo;
layout(binding = 2) uniform sampler2D gNormal;
layout(binding = 3) uniform sampler2D gMaterial;
layout(binding = 4) uniform sampler2D gDepth;

layout(location = 0) in vec2 vUv;

layout(location = 0) out vec4 out_fragColor;

void main()
{
  const ivec2 pixel = ivec2(gl_FragCoord.xy);
  const float depth = texelFetch(gDepth, pixel, 0).r;

  // Nothing was drawn here: sky (reverse-Z, the depth buffer is cleared to 0)
  if (depth <= 0.0)
  {
    const int mode = int(params.debug.x + 0.5);
    out_fragColor = vec4(mode == DEBUG_VIEW_SHADED ? skyColorTonemapped() : vec3(0.0), 1.0);
    return;
  }

  GBufferTexel t;
  t.rt0 = texelFetch(gAlbedo, pixel, 0);
  t.rt1 = texelFetch(gNormal, pixel, 0);
  t.rt2 = texelFetch(gMaterial, pixel, 0);
  const Surface s = unpackGBuffer(t, params.gbuffer.x > 0.5);

  // Pixel center in [0,1]^2
  const vec2 uv = gl_FragCoord.xy * params.screen.zw;
  const vec3 wPos = reconstructWorldPos(uv, depth, params.invViewProjRel, params.cameraPos.xyz);

  out_fragColor = vec4(shadeSurface(s, wPos), 1.0);
}
