#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

// Deferred lighting (G-buffer resolve): one fullscreen triangle, every pixel reads its surface
// from the G-buffer, restores the world position from depth and runs the Cook-Torrance BRDF.
// Pixels without geometry show the baked sky cubemap in the view direction.

#include "SceneParams.h"

layout(binding = 0) uniform SceneUbo
{
  SceneParams params;
};

// Baked procedural sky (SkyBaker). Declared before shading.glsl, which uses it for ambient light
layout(binding = 5) uniform samplerCube skyCube;

#include "gbuffer.glsl"
#include "shading.glsl"

layout(binding = 1) uniform sampler2D gAlbedo;
layout(binding = 2) uniform sampler2D gNormal;
layout(binding = 3) uniform sampler2D gMaterial;
layout(binding = 4) uniform sampler2D gDepth;
// Raw depth of the sun shadow map. Reversed-Z: the fragment is lit when its light-space depth
// is greater than or equal to the recorded one.
layout(binding = 6) uniform sampler2D shadowMap;

layout(location = 0) in vec2 vUv;

layout(location = 0) out vec4 out_fragColor;

// Percentage-Closer Filtering over the sun shadow map. The kernel is square, its radius in
// texels comes from the GUI (2 -> the 5x5 kernel required by the task). All offsets are in
// light-space texture coordinates, the reference depth is the shaded fragment's own depth.
float sunShadowFactor(vec3 wPos, vec3 N)
{
  if (params.shadowOptions.y < 0.5)
    return 1.0;

  const vec3 L = normalize(params.sunDir.xyz);
  const float NdotL = clamp(dot(N, L), 0.0, 1.0);
  const float slope = 1.0 - NdotL;

  // Move the lookup point along the normal towards the sun: on grazing surfaces a whole
  // texel of the shadow map covers a long stretch of the surface, so the depth of the
  // fragment and of the recorded surface differ by much more than on a face-on surface.
  const vec3 p = wPos + N * (params.shadow.z + 0.15 * slope);

  const vec4 clip = params.shadowViewProj * vec4(p, 1.0);
  const vec3 ndc = clip.xyz / clip.w;
  const vec2 uv = ndc.xy * 0.5 + 0.5;

  // Outside of the shadow map: fully lit (the ortho frustum should cover the whole terrain)
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)
    return 1.0;

  // Constant + slope-scaled depth bias, converted from meters to the normalized light depth.
  // Reversed-Z, so the bias is ADDED to the reference depth.
  const float ref = ndc.z + (0.02 + params.shadow.w * slope) * params.shadow.y;

  const int radius = int(params.shadowOptions.x + 0.5);
  float sum = 0.0;
  float count = 0.0;
  for (int y = -4; y <= 4; ++y)
    for (int x = -4; x <= 4; ++x)
    {
      if (abs(x) > radius || abs(y) > radius)
        continue;
      const float recorded = texture(shadowMap, uv + vec2(x, y) * params.shadow.x).r;
      sum += ref >= recorded ? 1.0 : 0.0;
      count += 1.0;
    }
  return sum / count;
}

void main()
{
  const ivec2 pixel = ivec2(gl_FragCoord.xy);
  const float depth = texelFetch(gDepth, pixel, 0).r;

  // Pixel center in [0,1]^2
  const vec2 uv = gl_FragCoord.xy * params.screen.zw;

  // Nothing was drawn here: sky (reverse-Z, the depth buffer is cleared to 0)
  if (depth <= 0.0)
  {
    const int mode = int(params.debug.x + 0.5);
    // Any point on the pixel's ray works: take the near plane (depth = 1 with reverse-Z).
    // The inverse matrix has no camera translation, so the result is the view direction itself.
    const vec4 rel = params.invViewProjRel * vec4(uv * 2.0 - 1.0, 1.0, 1.0);
    const vec3 dir = normalize(rel.xyz / rel.w);
    out_fragColor = vec4(mode == DEBUG_VIEW_SHADED ? skyRadiance(dir) : vec3(0.0), 1.0);
    return;
  }

  GBufferTexel t;
  t.rt0 = texelFetch(gAlbedo, pixel, 0);
  t.rt1 = texelFetch(gNormal, pixel, 0);
  t.rt2 = texelFetch(gMaterial, pixel, 0);
  const Surface s = unpackGBuffer(t, params.gbuffer.x > 0.5);

  const vec3 wPos = reconstructWorldPos(uv, depth, params.invViewProjRel, params.cameraPos.xyz);

  const float sunShadow = sunShadowFactor(wPos, s.normal);
  if (int(params.debug.x + 0.5) == DEBUG_VIEW_SHADOW)
  {
    out_fragColor = vec4(vec3(sunShadow), 1.0);
    return;
  }

  out_fragColor = vec4(shadeSurface(s, wPos, sunShadow), 1.0);
}
