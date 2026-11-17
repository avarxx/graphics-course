#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

// Projective decals: run after the G-buffer is filled and before lighting. For every pixel
// covered by a decal box the world position is restored from depth and moved into the box's
// local space. If it is inside, the decal textures are sampled and blended (with hardware alpha
// blending) into the albedo, normal and roughness channels of the G-buffer.

#include "decal_common.glsl"
#include "gbuffer.glsl"

layout(binding = 2) uniform sampler2D gDepth;
// Decal texture arrays, one layer per decal kind
layout(binding = 3) uniform sampler2DArray decalAlbedo;    // rgb = albedo (sRGB), a = mask
layout(binding = 4) uniform sampler2DArray decalNormal;    // rgb = tangent space normal, a = roughness

layout(location = 0) flat in uint vDecal;

// Same attachments as the G-buffer pass. Every output's alpha is the blend factor for that target.
layout(location = 0) out vec4 out_albedo;
layout(location = 1) out vec4 out_normal;
layout(location = 2) out vec4 out_material;

vec3 worldPosAt(ivec2 pixel)
{
  const ivec2 p = clamp(pixel, ivec2(0), ivec2(params.screen.xy) - 1);
  const float depth = texelFetch(gDepth, p, 0).r;
  const vec2 uv = (vec2(p) + 0.5) * params.screen.zw;
  return reconstructWorldPos(uv, depth, params.invViewProjRel, params.cameraPos.xyz);
}

// Of the two neighbours along an axis, take the one that is closer in depth.
// This avoids smearing derivatives across silhouettes (e.g. rock edges over terrain).
vec3 bestDelta(vec3 center, vec3 minus, vec3 plus)
{
  const vec3 a = center - minus;
  const vec3 b = plus - center;
  return dot(a, a) < dot(b, b) ? a : b;
}

void main()
{
  const ivec2 pixel = ivec2(gl_FragCoord.xy);
  const float depth = texelFetch(gDepth, pixel, 0).r;
  if (depth <= 0.0) // sky (reverse-Z)
    discard;

  const DecalGpu decal = decals[vDecal];

  const vec2 uv = gl_FragCoord.xy * params.screen.zw;
  const vec3 wPos = reconstructWorldPos(uv, depth, params.invViewProjRel, params.cameraPos.xyz);
  const vec3 local = (decal.worldToBox * vec4(wPos, 1.0)).xyz;
  if (any(greaterThan(abs(local), vec3(0.5))))
    discard;

  // Surface frame from the depth buffer: geometric normal + world-space pixel footprint
  const vec3 dx = bestDelta(wPos, worldPosAt(pixel - ivec2(1, 0)), worldPosAt(pixel + ivec2(1, 0)));
  const vec3 dy = bestDelta(wPos, worldPosAt(pixel - ivec2(0, 1)), worldPosAt(pixel + ivec2(0, 1)));
  vec3 N = normalize(cross(dx, dy));
  if (dot(N, params.cameraPos.xyz - wPos) < 0.0)
    N = -N;

  // Box axes in world space (columns of boxToWorld), normalized
  const vec3 axisX = normalize(decal.boxToWorld[0].xyz);
  const vec3 axisY = normalize(decal.boxToWorld[1].xyz);
  const vec3 axisZ = normalize(decal.boxToWorld[2].xyz);

  // Fade out on surfaces that are not facing the projection direction (would stretch)
  // and near the top/bottom faces of the box
  const float facing = dot(N, axisY);
  const float fadeStart = decal.params1.y;
  float fade = smoothstep(fadeStart, fadeStart + 0.25, facing);
  fade *= 1.0 - smoothstep(0.4, 0.5, abs(local.y));

  // Texture coordinates and their screen-space gradients (analytic, so mip selection does not
  // break at depth discontinuities the way dFdx/dFdy would)
  const vec2 texUv = local.xz + 0.5;
  const mat3 toBox = mat3(decal.worldToBox);
  const vec2 gradX = (toBox * dx).xz;
  const vec2 gradY = (toBox * dy).xz;
  const vec3 texCoord = vec3(texUv, decal.params0.x);

  const vec4 albedoMask = textureGrad(decalAlbedo, texCoord, gradX, gradY);
  const vec4 normalRough = textureGrad(decalNormal, texCoord, gradX, gradY);
  vec3 nTs = normalRough.xyz * 2.0 - 1.0;
  const float roughness = normalRough.a;

  const float alpha = clamp(albedoMask.a * decal.params0.y * pc.global.x * fade, 0.0, 1.0);

  // Tangent frame on the surface, aligned with the decal's u/v directions
  const vec3 T = normalize(axisX - N * dot(N, axisX));
  const vec3 B = normalize(axisZ - N * dot(N, axisZ) - T * dot(T, axisZ));
  const float normalIntensity = decal.params0.w;
  nTs.xy *= normalIntensity;
  const vec3 decalN = normalize(T * nTs.x + B * nTs.y + N * max(nTs.z, 1e-3));

  Surface s;
  s.albedo = albedoMask.rgb * decal.tint.rgb;
  s.normal = decalN;
  s.roughness = roughness;
  s.metallic = 0.0;
  s.ao = 1.0;
  const GBufferTexel t = packGBuffer(s, pc.global.z > 0.5);

  // Write masks in the pipeline make sure only albedo.rgb, normal and roughness are touched
  out_albedo = vec4(t.rt0.rgb, alpha * decal.params0.z);
  out_normal = vec4(t.rt1.xyz, alpha * min(normalIntensity, 1.0));
  out_material = vec4(t.rt2.r, 0.0, 0.0, alpha * decal.params1.x);

  // Debug: outline of the projected box
  if (pc.global.y > 0.5)
  {
    const vec2 edge = 0.5 - abs(local.xz);
    const vec2 w = vec2(length(gradX), length(gradY));
    if (min(edge.x, edge.y) < 2.0 * max(w.x, w.y))
      out_albedo = vec4(1.0, 0.15, 0.05, 1.0);
  }
}
