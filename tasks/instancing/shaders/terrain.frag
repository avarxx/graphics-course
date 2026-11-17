#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "TerrainParams.h"
#include "terrain_common.glsl"
#include "pbr.glsl"


layout(location = 0) out vec4 out_fragColor;

layout(location = 0) in VS_OUT
{
  vec3 wPos;
  vec2 uv;
} surf;

layout(binding = 0) uniform TerrainUbo
{
  TerrainParams params;
};

layout(binding = 1) uniform sampler2D heightMap;
// Layers: 0 = sand, 1 = grass, 2 = rock, 3 = snow
layout(binding = 2) uniform sampler2DArray albedoArray;
// R = roughness, G = height (displacement)
layout(binding = 3) uniform sampler2DArray roughHeightArray;

const int LAYER_COUNT = 4;

void main()
{
  const float worldSize = params.terrainSize.x;
  const float heightScale = params.terrainSize.y;
  const float seaLevel = params.terrainSize.z;

  // --- Geometry ---------------------------------------------------------------------------------
  const float hNorm = sampleHeight(heightMap, surf.uv);
  const vec3 geomN = terrainNormal(heightMap, surf.uv, worldSize, heightScale);

  // --- Layer weights from altitude and slope ----------------------------------------------------
  const vec4 t = params.layerHeights;
  const float bw = max(t.w, 1e-3);
  const float s0 = smoothstep(t.x - bw, t.x + bw, hNorm);
  const float s1 = smoothstep(t.y - bw, t.y + bw, hNorm);
  const float s2 = smoothstep(t.z - bw, t.z + bw, hNorm);

  float w[LAYER_COUNT];
  w[0] = 1.0 - s0;
  w[1] = s0 - s1;
  w[2] = s1 - s2;
  w[3] = s2;

  // Steep slopes are rocky: snow and grass do not stay there
  const float slope = 1.0 - geomN.y;
  const float rocky = smoothstep(0.18, 0.38, slope) * params.detail.z;
  const float moved = (w[1] + w[3] * 0.8 + w[0] * 0.4) * rocky;
  w[1] *= 1.0 - rocky;
  w[3] *= 1.0 - 0.8 * rocky;
  w[0] *= 1.0 - 0.4 * rocky;
  w[2] += moved;

  // --- Material sampling ------------------------------------------------------------------------
  const vec2 texUv = surf.wPos.xz / params.detail.x;
  const float du = 1.0 / float(textureSize(roughHeightArray, 0).x);

  vec3 albedo[LAYER_COUNT];
  float rough[LAYER_COUNT];
  float disp[LAYER_COUNT];
  vec2 dispGrad[LAYER_COUNT];
  for (int i = 0; i < LAYER_COUNT; ++i)
  {
    albedo[i] = texture(albedoArray, vec3(texUv, i)).rgb;
    const vec2 rh = texture(roughHeightArray, vec3(texUv, i)).rg;
    rough[i] = rh.r;
    disp[i] = rh.g;
    const float hx = texture(roughHeightArray, vec3(texUv + vec2(du, 0), i)).g;
    const float hy = texture(roughHeightArray, vec3(texUv + vec2(0, du), i)).g;
    dispGrad[i] = vec2(hx - rh.g, hy - rh.g);
  }

  // Height-based blending: the layer whose displacement "sticks out" wins at transitions
  const float sharp = params.detail.w;
  float wSum = 0.0;
  for (int i = 0; i < LAYER_COUNT; ++i)
  {
    w[i] = max(w[i], 0.0) * pow(disp[i] + 0.05, sharp);
    wSum += w[i];
  }
  for (int i = 0; i < LAYER_COUNT; ++i)
    w[i] /= max(wSum, 1e-5);

  vec3 baseColor = vec3(0.0);
  float roughness = 0.0;
  vec2 grad = vec2(0.0);
  for (int i = 0; i < LAYER_COUNT; ++i)
  {
    baseColor += w[i] * albedo[i];
    roughness += w[i] * rough[i];
    grad += w[i] * dispGrad[i];
  }
  roughness = clamp(roughness * params.debug.z, 0.04, 1.0);

  // --- Detail normal from the Height maps (bump mapping, no geometry change) --------------------
  vec3 N = geomN;
  if (params.debug.y > 0.5)
  {
    // Height texture gradient in "meters of bump per meter of surface"
    const float bumpDepth = 0.06 * params.detail.y;
    const vec2 slopeXZ = grad * bumpDepth / (du * params.detail.x);
    vec3 T = normalize(vec3(1, 0, 0) - geomN * geomN.x);
    vec3 B = normalize(vec3(0, 0, 1) - geomN * geomN.z);
    N = normalize(geomN - slopeXZ.x * T - slopeXZ.y * B);
  }

  // --- Lighting ---------------------------------------------------------------------------------
  const vec3 V = normalize(params.cameraPos.xyz - surf.wPos);
  const vec3 L = normalize(params.sunDir.xyz);
  const vec3 radiance = params.sunColor.rgb * params.sunDir.w;

  vec3 color = shadeDirectional(baseColor, roughness, 0.0, N, V, L, radiance);
  color += shadeAmbient(
    baseColor, 1.0, N, params.skyColor.rgb, params.groundColor.rgb, params.sunColor.w);

  // Underwater tint: the island is surrounded by the sea
  if (hNorm < seaLevel)
  {
    const float depth = (seaLevel - hNorm) * heightScale;
    const vec3 waterColor = vec3(0.02, 0.12, 0.18) * (params.sunDir.w * 0.3 + params.sunColor.w);
    const float k = 1.0 - exp(-depth * 0.35);
    color = mix(color * vec3(0.6, 0.85, 0.9), waterColor, clamp(k, 0.0, 0.97));
  }

  color = tonemapACES(color * params.skyColor.w);

  // Distance fog towards the (tonemapped) sky color
  const float dist = length(params.cameraPos.xyz - surf.wPos);
  const float fog = 1.0 - exp(-dist * params.groundColor.w);
  color = mix(color, tonemapACES(params.skyColor.rgb * params.skyColor.w), clamp(fog, 0.0, 1.0));

  // --- Debug views ------------------------------------------------------------------------------
  const int mode = int(params.debug.x + 0.5);
  if (mode == 1)
    color = N * 0.5 + 0.5;
  else if (mode == 2)
    color = geomN * 0.5 + 0.5;
  else if (mode == 3)
    color = vec3(w[0] + w[3], w[1] + w[3], w[2] + w[3]);
  else if (mode == 4)
    color = vec3(hNorm);

  out_fragColor = vec4(color, 1.0);
}
