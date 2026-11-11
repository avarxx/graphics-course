#ifndef SHADING_GLSL_INCLUDED
#define SHADING_GLSL_INCLUDED

// Lighting of a single surface point. Used by the deferred lighting pass and by the forward
// reference path, so both produce exactly the same image. Expects `params` (SceneParams).

#include "surface.glsl"
#include "pbr.glsl"

vec3 skyColorTonemapped()
{
  return tonemapACES(params.skyColor.rgb * params.skyColor.w);
}

vec3 shadeSurface(Surface s, vec3 wPos)
{
  const vec3 V = normalize(params.cameraPos.xyz - wPos);
  const vec3 L = normalize(params.sunDir.xyz);
  const vec3 radiance = params.sunColor.rgb * params.sunDir.w;

  // Cook-Torrance (GGX) for the sun + hemispherical ambient
  vec3 color = shadeDirectional(s.albedo, s.roughness, s.metallic, s.normal, V, L, radiance);
  color += shadeAmbient(
    s.albedo * (1.0 - s.metallic), s.ao, s.normal, params.skyColor.rgb, params.groundColor.rgb,
    params.sunColor.w);
  color += params.lighting.x * shadeAmbientSpecular(
    s.albedo, s.roughness, s.metallic, s.ao, s.normal, V,
    params.skyColor.rgb, params.groundColor.rgb, params.sunColor.w);

  // Underwater tint: the island is surrounded by the sea
  const float seaHeight = params.terrainSize.z * params.terrainSize.y;
  if (wPos.y < seaHeight)
  {
    const float depth = seaHeight - wPos.y;
    const vec3 waterColor = vec3(0.02, 0.12, 0.18) * (params.sunDir.w * 0.3 + params.sunColor.w);
    const float k = 1.0 - exp(-depth * 0.35);
    color = mix(color * vec3(0.6, 0.85, 0.9), waterColor, clamp(k, 0.0, 0.97));
  }

  color = tonemapACES(color * params.skyColor.w);

  // Distance fog towards the (tonemapped) sky color
  const float dist = length(params.cameraPos.xyz - wPos);
  const float fog = 1.0 - exp(-dist * params.groundColor.w);
  color = mix(color, skyColorTonemapped(), clamp(fog, 0.0, 1.0));

  // --- Debug views ------------------------------------------------------------------------------
  const int mode = int(params.debug.x + 0.5);
  if (mode == DEBUG_VIEW_ALBEDO || mode == DEBUG_VIEW_LAYERS || mode == DEBUG_VIEW_HEIGHT)
    color = s.albedo;
  else if (mode == DEBUG_VIEW_NORMAL)
    color = s.normal * 0.5 + 0.5;
  else if (mode == DEBUG_VIEW_ROUGHNESS)
    color = vec3(s.roughness);
  else if (mode == DEBUG_VIEW_METALLIC)
    color = vec3(s.metallic);
  else if (mode == DEBUG_VIEW_AO)
    color = vec3(s.ao);
  else if (mode == DEBUG_VIEW_DEPTH)
    color = vec3(1.0 - exp(-dist / 250.0));
  else if (mode == DEBUG_VIEW_POSITION)
  {
    // 10 m world-space grid: lines must stay glued to the surface while the camera moves
    const vec3 cell = fract(wPos / 10.0);
    const vec3 line = smoothstep(vec3(0.0), vec3(0.04), cell) * smoothstep(vec3(0.0), vec3(0.04), 1.0 - cell);
    color = mix(vec3(1.0, 0.85, 0.2), fract(wPos / 100.0), min(line.x, min(line.y, line.z)));
  }

  return color;
}

#endif // SHADING_GLSL_INCLUDED
